# src/pipeline/training_pipeline.py
import os
import sys
import logging
import argparse # For command-line arguments
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn # Using generic sklearn logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer # Import the new trainer
from src.utils import load_config, load_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingPipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = load_config()
        self.params = load_params()
        self.mlflow_config = self.config['mlflow_config']

    def run(self):
        """Execute the full training pipeline for the specified model."""
        try:
            # --- 1. Set up MLflow ---
            mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
            mlflow.set_experiment(self.mlflow_config['experiment_name'])
            
            with mlflow.start_run(run_name=f"{self.model_name}-run") as run:
                logging.info(f"Started MLflow Run for model '{self.model_name}' with ID: {run.info.run_id}")
                mlflow.set_tag("model_name", self.model_name) # Tag the run with the model name

                # --- 2. Data Loading and Preprocessing ---
                df = self._load_data()
                X_train, y_train, X_test, y_test, preprocessor, feature_names = self._preprocess_data(df)
                
                # --- 3. Log Parameters and Artifacts ---
                mlflow.log_params(self.params['train'])
                if self.params.get(self.model_name):
                    mlflow.log_params(self.params[self.model_name])
                
                # Log configs and preprocessor
                mlflow.log_artifact("config/config.yaml")
                mlflow.log_artifact("params.yaml")
                joblib.dump(preprocessor, "preprocessor.joblib")
                mlflow.log_artifact("preprocessor.joblib", artifact_path="preprocessor")

                # --- 4. Model Training ---
                trainer = ModelTrainer(model_name=self.model_name, params=self.params)
                model = trainer.train(X_train, y_train)

                # --- 5. Model Evaluation and Logging ---
                self._evaluate_and_log(model, X_test, y_test)

                # --- 6. Model Registration ---
                self._register_model(model, X_train[:5]) # Pass a sample for signature inference

        except Exception as e:
            logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
            raise

    # _load_data and _preprocess_data methods remain unchanged...
    def _load_data(self):
        """Loads data from the source specified in config."""
        raw_data_path = os.path.join(
            self.config['data_source']['raw_data_dir'], 
            self.config['data_source']['raw_data_filename']
        )
        logging.info(f"Loading raw data from {raw_data_path}")
        return pd.read_csv(raw_data_path)

    def _preprocess_data(self, df):
        """Splits and preprocesses data."""
        logging.info("Splitting and preprocessing data.")
        train_params = self.params['train']
        
        train_df, test_df = train_test_split(
            df,
            test_size=train_params['test_size'],
            random_state=train_params['random_state'],
            stratify=df[train_params['target_column']]
        )
        
        transformation_config = DataTransformationConfig(target_column=train_params['target_column'])
        data_transformer = DataTransformation(config=transformation_config)
        
        X_train_proc, y_train, X_test_proc, y_test, preprocessor, f_names = \
            data_transformer.initiate_data_transformation(train_df, test_df)
        
        return X_train_proc, y_train, X_test_proc, y_test, preprocessor, f_names

    def _evaluate_and_log(self, model, X_test, y_test):
        """Evaluates the model and logs metrics and plots to MLflow."""
        logging.info("Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_pr": average_precision_score(y_test, y_pred_proba),
            "auc_roc": roc_auc_score(y_test, y_pred_proba)
        }
        
        mlflow.log_metrics(metrics)
        logging.info(f"Logged Metrics: {metrics}")

        # Create and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {self.model_name}')
        
        confusion_matrix_path = "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        mlflow.log_artifact(confusion_matrix_path, "plots")
        plt.close()

    def _register_model(self, model, input_example):
        """Logs and registers the model in the MLflow Model Registry."""
        # Use the generic mlflow.sklearn.log_model
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
    args = parser.parse_args()

    pipeline = TrainingPipeline(model_name=args.model)
    pipeline.run()