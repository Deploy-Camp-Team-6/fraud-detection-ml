import os
import sys
import logging
import argparse
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import load_config

def predict(model_name: str, alias: str, input_path: str, output_path: str):
    """Load a registered model from MLflow and generate predictions.

    Args:
        model_name (str): Base name of the model in the registry (e.g., 'xgboost').
        alias (str): The model alias to load (e.g., 'champion').
        input_path (str): Path to the input CSV data.
        output_path (str): Path to save the predictions CSV.
    """
    try:
        config = load_config()
        mlflow_config = config['mlflow_config']
        registry_uri = mlflow_config.get('registry_uri', mlflow_config['tracking_uri'])
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_registry_uri(registry_uri)

        # Resolve the alias to a concrete model version
        registered_model_name = (
            f"{mlflow_config['registered_model_base_name']}-{model_name}"
        )
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(
            name=registered_model_name, alias=alias
        )
        model_uri = f"models:/{registered_model_name}/{model_version.version}"
        logging.info(f"Loading model from URI: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)

        # Load data for prediction
        logging.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

        # The loaded model is a pipeline, so it will handle preprocessing automatically
        logging.info("Making predictions...")
        predictions = model.predict(df)

        # Add predictions to the dataframe
        df['prediction'] = predictions

        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logging.info(f"Saving predictions to {output_path}")
            df.to_csv(output_path, index=False)
        else:
            logging.info("Predictions:")
            print(df.to_string())

        logging.info("Prediction complete.")

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Make predictions using a registered model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["xgboost", "lightgbm", "logistic_regression"],
        help="Base name of the model.",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=os.getenv("MODEL_REGISTRY_ALIAS", "champion"),
        help="Alias of the model version to use.",
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output CSV. If not provided, prints to console.",
    )

    args = parser.parse_args()
    predict(model_name=args.model, alias=args.alias, input_path=args.input, output_path=args.output)
