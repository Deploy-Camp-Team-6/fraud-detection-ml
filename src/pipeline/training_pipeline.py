# src/pipeline/training_pipeline.py
import os
import sys
import logging
import argparse
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.components.model_trainer import ModelTrainer
from src.utils import load_config, load_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingPipeline:
    def __init__(self, model_name: str, train_data_path: str, test_data_path: str, preprocessor_path: str):
        self.model_name = model_name
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.preprocessor_path = preprocessor_path

        self.config = load_config()
        self.params = load_params()
        self.mlflow_config = self.config['mlflow_config']
        self.target_column = self.params['train']['target_column']

    def run(self):
        """Execute the full training pipeline using preprocessed data."""
        try:
            # --- 1. Load Preprocessed Data and Preprocessor ---
            logging.info("Loading preprocessed data and preprocessor artifact.")
            train_df = pd.read_csv(self.train_data_path)
            test_df = pd.read_csv(self.test_data_path)
            preprocessor = joblib.load(self.preprocessor_path)

            # Separate features and target
            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            # --- 2. Set up MLflow ---
            mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
            mlflow.set_experiment(self.mlflow_config['experiment_name'])
            
            run_name_suffix = self.config['mlflow_artifacts']['run_name_suffix']
            with mlflow.start_run(run_name=f"{self.model_name}{run_name_suffix}") as run:
                logging.info(f"Started MLflow Run for model '{self.model_name}' with ID: {run.info.run_id}")
                mlflow.set_tag("model_name", self.model_name)

                # --- 3. Log Parameters and Artifacts ---
                mlflow.log_params(self.params['train'])
                if self.params.get(self.model_name):
                    mlflow.log_params(self.params[self.model_name])
                
                # Log configs and the preprocessor artifact from its path
                mlflow.log_artifact("config/config.yaml")
                mlflow.log_artifact("params.yaml")
                preprocessor_artifact_path = self.config['mlflow_artifacts']['preprocessor_path']
                mlflow.log_artifact(self.preprocessor_path, artifact_path=preprocessor_artifact_path)

                # --- 4. Model Training ---
                trainer = ModelTrainer(model_name=self.model_name, params=self.params)
                model = trainer.train(X_train, y_train)

                # --- 5. Model Evaluation and Logging ---
                self._evaluate_and_log(model, X_test, y_test)

                # --- 6. Model Registration ---
                self._register_model(model, X_train.head(5)) # Pass a sample for signature inference

        except Exception as e:
            logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)
            raise

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

        # Use labels from config
        labels = self.config['plotting']['confusion_matrix_labels']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {self.model_name}')
        
        # Define artifact path from config
        plot_filename = self.config['artifacts']['confusion_matrix_path']
        plt.savefig(plot_filename)

        # Use artifact path from config
        plot_artifact_path = self.config['mlflow_artifacts']['plot_path']
        mlflow.log_artifact(plot_filename, plot_artifact_path)
        plt.close()

    def _register_model(self, model, input_example):
        """Logs and registers the model in the MLflow Model Registry."""
        logging.info("Logging and registering the model using mlflow.sklearn.")
        
        registered_model_name = f"{self.mlflow_config['registered_model_base_name']}-{self.model_name}"
        model_artifact_path = self.config['mlflow_artifacts']['model_path']
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_artifact_path,
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
    parser.add_argument("--train-data-path", type=str, required=True, help="Path to the preprocessed training data CSV.")
    parser.add_argument("--test-data-path", type=str, required=True, help="Path to the preprocessed test data CSV.")
    parser.add_argument("--preprocessor-path", type=str, required=True, help="Path to the fitted preprocessor artifact.")

    args = parser.parse_args()

    pipeline = TrainingPipeline(
        model_name=args.model,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        preprocessor_path=args.preprocessor_path,
    )
    pipeline.run()