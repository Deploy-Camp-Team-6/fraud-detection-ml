import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import boto3
from botocore.exceptions import ClientError
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import load_config, load_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingPipeline:
    def __init__(self, model_name: str, tune: bool):
        self.model_name = model_name
        self.tune = tune
        self.config = load_config()
        self.params = load_params()
        self.mlflow_config = self.config['mlflow_config']
        self.cv_splitter = StratifiedKFold(
            n_splits=self.params['train']['n_splits'],
            shuffle=True,
            random_state=self.params['train']['random_state']
        )

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
            # If the bucket does not exist, create it
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

    def run(self):
        """Execute the full training or tuning pipeline."""
        run_name = f"{self.model_name}-{'tuning' if self.tune else 'cv-evaluation'}"
        try:
            # First, ensure the artifact bucket exists.
            self._ensure_mlflow_bucket_exists()

            mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
            mlflow.set_experiment(self.mlflow_config['experiment_name'])
            
            with mlflow.start_run(run_name=run_name) as run:
                logging.info(f"Started MLflow Run '{run_name}' with ID: {run.info.run_id}")
                mlflow.set_tag("model_name", self.model_name)
                mlflow.set_tag("tuning_run", str(self.tune))

                # --- 1. Load Data ---
                df = self._load_data()
                X = df.drop(columns=[self.config['features']['target_column']])
                y = df[self.config['features']['target_column']]

                # --- 2. Create Preprocessing and Model Pipeline ---
                data_transformer = DataTransformation(
                    feature_config=self.config['features'],
                    params=self.params
                )
                preprocessor = data_transformer.preprocessor

                model_trainer = ModelTrainer(model_name=self.model_name, params=self.params)
                # Pass y for imbalance calculation
                model = model_trainer._get_model(y_train=y)

                full_pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", model)
                ])

                # --- 3. Run Tuning or Cross-Validation ---
                if self.tune:
                    self._run_tuning(full_pipeline, X, y)
                else:
                    self._run_cross_validation(full_pipeline, X, y)

        except Exception as e:
            logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
            raise

    def _load_data(self):
        """Loads data from the source specified in config."""
        raw_data_path = os.path.join(
            self.config['data_source']['raw_data_dir'], 
            self.config['data_source']['raw_data_filename']
        )
        logging.info(f"Loading raw data from {raw_data_path}")
        return pd.read_csv(raw_data_path)

    def _run_tuning(self, pipeline, X, y):
        """Performs hyperparameter tuning using GridSearchCV."""
        logging.info(f"Starting hyperparameter tuning for {self.model_name}...")
        
        param_grid = self.params['tuning'][self.model_name]['param_grid']
        
        # Using F1 score for tuning as it's a good metric for imbalanced datasets
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=self.cv_splitter,
            scoring='f1',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X, y)

        logging.info(f"Best parameters found: {grid_search.best_params_}")
        logging.info(f"Best F1 score from tuning: {grid_search.best_score_:.4f}")

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_tuning_f1_score", grid_search.best_score_)

        # Log the best model found during the search
        self._log_and_register_model(grid_search.best_estimator_, X.head(5))

    def _run_cross_validation(self, pipeline, X, y):
        """Performs cross-validation and logs results."""
        logging.info(f"Starting {self.params['train']['n_splits']}-fold cross-validation for {self.model_name}...")

        scoring = {
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score)
        }
        
        scores = cross_validate(pipeline, X, y, cv=self.cv_splitter, scoring=scoring, n_jobs=-1)

        # Log average metrics
        avg_metrics = {f"avg_cv_{metric}": np.mean(values) for metric, values in scores.items()}
        std_metrics = {f"std_cv_{metric}": np.std(values) for metric, values in scores.items()}

        mlflow.log_metrics(avg_metrics)
        mlflow.log_metrics(std_metrics)
        
        logging.info("Cross-validation metrics:")
        for metric, value in avg_metrics.items():
            logging.info(f"  {metric}: {value:.4f}")

        # Train final model on all data and log it
        logging.info("Training final model on all data...")
        pipeline.fit(X, y)
        self._log_and_register_model(pipeline, X.head(5))

    def _log_and_register_model(self, model, input_example):
        """Logs and registers the model in the MLflow Model Registry."""
        logging.info("Logging and registering the model using mlflow.sklearn.")
        
        registered_model_name = f"{self.mlflow_config['registered_model_base_name']}-{self.model_name}"
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
            input_example=input_example
        )
        logging.info(f"Model registered as '{registered_model_name}'")

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
        "--tune",
        action="store_true",
        help="If set, run hyperparameter tuning instead of single cross-validation."
    )
    args = parser.parse_args()

    pipeline = TrainingPipeline(model_name=args.model, tune=args.tune)
    pipeline.run()
