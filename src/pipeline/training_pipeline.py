import os
import sys
import logging
import argparse
import copy
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import boto3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from botocore.exceptions import ClientError
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import load_config, load_params, drop_constant_columns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingPipeline:
    def __init__(self, model_name: str, tune: bool, use_best_params: bool = False, tune_and_evaluate: bool = False, model_alias: str | None = None):
        self.model_name = model_name
        self.tune = tune
        self.use_best_params = use_best_params
        self.tune_and_evaluate = tune_and_evaluate
        # Use model aliases instead of stages when registering models
        self.model_alias = model_alias or os.getenv("MODEL_REGISTRY_ALIAS", "champion")
        self.config = load_config()
        self.params = load_params()
        self.mlflow_config = self.config['mlflow_config']
        self.cv_splitter = StratifiedKFold(
            n_splits=self.params['train']['n_splits'],
            shuffle=True,
            random_state=self.params['train']['random_state']
        )
        self.project_root = Path(__file__).resolve().parents[2]

    def _ensure_mlflow_bucket_exists(self):
        """Checks if the MLflow artifact bucket exists in MinIO/S3 and creates it if not."""
        endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        bucket_name = self.config['minio_credentials']['bucket_name']

        if not all([endpoint_url, access_key, secret_key, bucket_name]):
            logging.warning(
                "One or more S3 environment variables are not set. Skipping bucket creation check. "
                "This is expected for local runs without MinIO."
            )
            return

        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1' # Default region, can be anything for MinIO
        )

        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logging.info(f"S3 Bucket '{bucket_name}' already exists.")
        except ClientError as e:
            if e.response['Error']['Code'] == '404' or 'NoSuchBucket' in str(e):
                logging.info(f"S3 Bucket '{bucket_name}' not found. Attempting to create it.")
                try:
                    s3_client.create_bucket(Bucket=bucket_name)
                    logging.info(f"S3 Bucket '{bucket_name}' created successfully.")
                except ClientError as create_error:
                    logging.error(f"Fatal: Failed to create S3 bucket '{bucket_name}'. Error: {create_error}")
                    raise
            else:
                logging.error(f"Fatal: An unexpected error occurred while checking for bucket '{bucket_name}'. Error: {e}")
                raise

    def _initialize_mlflow(self, params_to_log: dict):
        """Sets up the MLflow tracking URI, experiment, and base tags."""
        if self.tune_and_evaluate:
            run_type = "Tune-and-Evaluate"
        elif self.tune:
            run_type = "Tuning"
        elif self.use_best_params:
            run_type = "Best-Params-CV"
        else:
            run_type = "Default-Params-CV"

        run_name = f"{self.model_name}-{run_type.lower().replace('_', '-')}"

        self._ensure_mlflow_bucket_exists()
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['experiment_name'])

        # Start the parent run
        run = mlflow.start_run(run_name=run_name)
        logging.info(f"Started MLflow Run '{run_name}' with ID: {run.info.run_id}")

        # Set common tags
        mlflow.set_tag("model_name", self.model_name)
        mlflow.set_tag("run_type", run_type)

        # Log configuration and parameters
        self._log_config_and_params(params_to_log)

        return run

    def _log_config_and_params(self, params_to_log: dict):
        """Logs key-value parameters and configuration files as artifacts."""
        logging.info("Logging configuration and parameters to MLflow.")
        # Log all model parameters from the provided dict
        mlflow.log_params(params_to_log[self.model_name])
        # Log training parameters
        mlflow.log_params(params_to_log['train'])

        # Log config and params files as artifacts
        mlflow.log_artifact(self.project_root / "config/config.yaml", "config")
        mlflow.log_artifact(self.project_root / "params.yaml", "config")

    def _create_pipeline(self, params, y_data):
        """Creates a full scikit-learn pipeline with preprocessing and a model."""
        data_transformer = DataTransformation(
            feature_config=self.config['features'],
            params=params
        )
        preprocessor = data_transformer.preprocessor

        model_trainer = ModelTrainer(model_name=self.model_name, params=params)
        model = model_trainer._get_model(y_train=y_data)

        handle_imbalance = params['train'].get('handle_imbalance', False)

        if handle_imbalance and self.model_name == "logistic_regression":
            return ImbPipeline([
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=params['train']['random_state'])),
                ("classifier", model)
            ])
        else:
            return SkPipeline([
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

    def run(self):
        """Execute the full training or tuning pipeline."""
        try:
            params_for_run = copy.deepcopy(self.params)

            if self.use_best_params:
                if 'best_params' not in self.params or self.model_name not in self.params['best_params']:
                    logging.error(f"Cannot use --use-best-params because 'best_params' section or model '{self.model_name}' not found in params.yaml.")
                    sys.exit(1)

                logging.info(f"Using 'best_params' for model '{self.model_name}'.")
                best_model_params = self.params['best_params'][self.model_name]
                params_for_run[self.model_name].update(best_model_params)

            self._initialize_mlflow(params_to_log=params_for_run)

            df = self._load_data()
            feature_cols = self.config['features']['numerical_cols'] + self.config['features']['categorical_cols']
            df, dropped = drop_constant_columns(df, feature_cols)
            if dropped:
                logging.warning(f"Dropping constant features: {dropped}")
                self.config['features']['numerical_cols'] = [c for c in self.config['features']['numerical_cols'] if c not in dropped]
                self.config['features']['categorical_cols'] = [c for c in self.config['features']['categorical_cols'] if c not in dropped]
            X = df.drop(columns=[self.config['features']['target_column']])
            y = df[self.config['features']['target_column']]

            initial_pipeline = self._create_pipeline(params_for_run, y)

            if self.tune_and_evaluate:
                logging.info("--- Starting Tuning Phase ---")
                best_params_from_tuning = self._run_tuning(initial_pipeline, X, y, register_model=False)

                logging.info("--- Tuning Complete. Starting Cross-Validation Phase ---")
                cleaned_params = {k.split('__', 1)[1]: v for k, v in best_params_from_tuning.items()}

                params_for_cv = copy.deepcopy(params_for_run)
                params_for_cv[self.model_name].update(cleaned_params)

                logging.info(f"Running CV with best parameters: {cleaned_params}")
                mlflow.log_params({f"final_best_{k}": v for k, v in cleaned_params.items()})

                cv_pipeline = self._create_pipeline(params_for_cv, y)
                self._run_cross_validation(cv_pipeline, X, y)

            elif self.tune:
                self._run_tuning(initial_pipeline, X, y, register_model=True)
            else:
                self._run_cross_validation(initial_pipeline, X, y)

        except Exception as e:
            logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
            raise
        finally:
            mlflow.end_run()
            logging.info("MLflow run finished.")

    def _load_data(self):
        """Loads data from the source specified in config."""
        raw_data_path = self.project_root / self.config['data_source']['raw_data_dir'] / self.config['data_source']['raw_data_filename']
        logging.info(f"Loading raw data from {raw_data_path}")
        return pd.read_csv(raw_data_path)

    def _run_tuning(self, pipeline, X, y, register_model: bool = True):
        """Performs hyperparameter tuning using GridSearchCV."""
        logging.info(f"Starting hyperparameter tuning for {self.model_name}...")
        
        param_grid = self.params['tuning'][self.model_name]['param_grid']
        
        # Log the parameter grid as a JSON artifact
        param_grid_path = self.project_root / "param_grid.json"
        with open(param_grid_path, 'w') as f:
            json.dump(param_grid, f, indent=4)
        mlflow.log_artifact(param_grid_path, "tuning")
        param_grid_path.unlink() # Clean up the file

        grid_search = GridSearchCV(
            estimator=pipeline, param_grid=param_grid,
            cv=self.cv_splitter, scoring='f1', n_jobs=-1, verbose=2
        )
        grid_search.fit(X, y)

        logging.info(f"Best parameters found: {grid_search.best_params_}")
        logging.info(f"Best F1 score from tuning: {grid_search.best_score_:.4f}")

        mlflow.log_params({f"tuning_best_{k}": v for k, v in grid_search.best_params_.items()})
        mlflow.log_metric("best_tuning_f1_score", grid_search.best_score_)

        # Log CV results as a CSV artifact
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cv_results_path = self.project_root / "tuning_cv_results.csv"
        cv_results_df.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path, "tuning")
        cv_results_path.unlink()

        # Log the best model found only if this is a standalone tuning run
        if register_model:
            self._log_and_register_model(grid_search.best_estimator_, X, y)

        return grid_search.best_params_

    def _run_cross_validation(self, pipeline, X, y):
        """Performs cross-validation with nested runs for each fold."""
        logging.info(f"Starting {self.params['train']['n_splits']}-fold cross-validation for {self.model_name}...")

        scoring = {
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score)
        }
        
        fold_scores = []
        for i, (train_index, test_index) in enumerate(self.cv_splitter.split(X, y)):
            with mlflow.start_run(nested=True, run_name=f"fold-{i+1}"):
                logging.info(f"--- Starting Fold {i+1}/{self.params['train']['n_splits']} ---")
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)

                fold_metrics = {
                    'precision': precision_score(y_test, preds, zero_division=0),
                    'recall': recall_score(y_test, preds),
                    'f1': f1_score(y_test, preds),
                    'roc_auc': roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
                }

                mlflow.log_metrics(fold_metrics)
                fold_scores.append(fold_metrics)
                logging.info(f"Fold {i+1} Metrics: {fold_metrics}")

        # Aggregate and log metrics to the parent run
        avg_metrics = {f"avg_cv_{metric}": np.mean([s[metric] for s in fold_scores]) for metric in scoring}
        std_metrics = {f"std_cv_{metric}": np.std([s[metric] for s in fold_scores]) for metric in scoring}

        mlflow.log_metrics(avg_metrics)
        mlflow.log_metrics(std_metrics)
        logging.info(f"Average CV Metrics: {avg_metrics}")

        # Train final model on all data and log it
        logging.info("Training final model on all data...")
        pipeline.fit(X, y)
        self._log_and_register_model(pipeline, X, y)

    def _log_and_register_model(self, model, X, y):
        """Logs artifacts, plots, and registers the model."""
        logging.info("Logging model, artifacts, and registering with MLflow.")
        
        # --- Log plots ---
        # Use out-of-fold predictions for an unbiased confusion matrix
        y_pred_cv = cross_val_predict(model, X, y, cv=self.cv_splitter, n_jobs=-1)
        self._log_confusion_matrix(y, y_pred_cv)

        # Log feature importance if available
        self._log_feature_importance(model)

        # --- Log and Register Model ---
        registered_model_name = f"{self.mlflow_config['registered_model_base_name']}-{self.model_name}"
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
            input_example=X.head(5)
        )

        client = MlflowClient()
        # If the model is registered, assign the provided alias
        if model_info.registered_model_version is not None:
            client.set_registered_model_alias(
                name=registered_model_name,
                version=model_info.registered_model_version,
                alias=self.model_alias,
            )
        logging.info(
            f"Model registered as '{registered_model_name}' and aliased as '{self.model_alias}'."
        )

    def _log_confusion_matrix(self, y_true, y_pred):
        """Creates, logs, and saves a confusion matrix plot."""
        try:
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

            # Log plot to MLflow
            mlflow.log_figure(fig, "artifacts/confusion_matrix.png")
            plt.close(fig)
            logging.info("Confusion matrix logged to MLflow.")
        except Exception as e:
            logging.warning(f"Could not generate or log confusion matrix: {e}")

    def _log_feature_importance(self, pipeline):
        """Extracts and logs feature importance plot if the model supports it."""
        if self.model_name not in ["xgboost", "lightgbm"]:
            logging.info(f"Feature importance plot not applicable for model '{self.model_name}'.")
            return

        try:
            # Extract the classifier and feature names from the pipeline
            classifier = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']

            # Get feature names after one-hot encoding
            categorical_features = preprocessor.named_transformers_['categorical'].get_feature_names_out(
                self.config['features']['categorical_cols']
            )
            num_features = self.config['features']['numerical_cols']
            feature_names = np.concatenate([num_features, categorical_features])

            importances = classifier.feature_importances_

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False).head(20) # Top 20

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
            ax.set_title(f"Feature Importance ({self.model_name})")
            plt.tight_layout()

            mlflow.log_figure(fig, "artifacts/feature_importance.png")
            plt.close(fig)
            logging.info("Feature importance plot logged to MLflow.")
        except Exception as e:
            logging.warning(f"Could not generate or log feature importance plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML training pipeline.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["xgboost", "lightgbm", "logistic_regression"],
        help="The model to train."
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default=os.getenv("MODEL_REGISTRY_ALIAS", "champion"),
        help=(
            "Alias name to assign to the registered model version. Defaults to the "
            "MODEL_REGISTRY_ALIAS environment variable or 'champion'."
        ),
    )

    # Mutually exclusive group for run modes
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning and log the best model."
    )
    mode_group.add_argument(
        "--use-best-params",
        action="store_true",
        help="Run cross-validation using the 'best_params' from params.yaml."
    )
    mode_group.add_argument(
        "--tune-and-evaluate",
        action="store_true",
        help="Run tuning, then immediately run CV with the best found params."
    )

    args = parser.parse_args()

    pipeline = TrainingPipeline(
        model_name=args.model,
        tune=args.tune,
        use_best_params=args.use_best_params,
        tune_and_evaluate=args.tune_and_evaluate,
        model_alias=args.model_alias
    )
    pipeline.run()
