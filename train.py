import argparse
import yaml
import mlflow
import pandas as pd
from src.data_utils import load_data, get_preprocessing_pipeline, split_data
from src.training_utils import train_and_evaluate, evaluate_on_test_set
import joblib
import os

def main(model_name: str):
    """Main training pipeline."""

    # Load configs
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name=f"{model_name}-run"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params['data_split'])
        mlflow.log_params(params['training'])

        # Load data
        df = load_data(config['data']['raw_path'])

        # Split data
        X_train, X_test, y_train, y_test = split_data(
            df,
            target_column=config['data']['target_column'],
            test_size=params['data_split']['test_size'],
            random_state=params['data_split']['random_state']
        )

        # Get preprocessing pipeline
        preprocessor = get_preprocessing_pipeline(
            numerical_cols=config['data']['numerical_columns'],
            categorical_cols=config['data']['categorical_columns'],
            log_transform_cols=['amount'] # As per feature engineering suggestion
        )

        # Fit preprocessor on training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Save the fitted preprocessor
        os.makedirs(config['artifacts']['model_dir'], exist_ok=True)
        preprocessor_path = os.path.join(config['artifacts']['model_dir'], config['artifacts']['preprocessor_name'])
        joblib.dump(preprocessor, preprocessor_path)
        mlflow.log_artifact(preprocessor_path, "preprocessor")

        # Train model
        best_model = train_and_evaluate(
            X_train_processed,
            y_train,
            model_name=model_name,
            params=params[model_name],
            cv_folds=params['training']['cv_folds'],
            imbalance_handler=params['training']['imbalance_handler'],
            random_state=params['data_split']['random_state']
        )

        # Evaluate on test set
        test_metrics = evaluate_on_test_set(best_model, X_test_processed, y_test)
        print(f"Test Set Metrics for {model_name}: {test_metrics}")

        # Log final model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=f"{config['mlflow']['registered_model_base_name']}-{model_name}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['xgboost', 'lightgbm', 'logistic_regression'],
        help="Model to train."
    )
    args = parser.parse_args()
    main(args.model)
