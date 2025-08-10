import os
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Ensure src is on path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.training_pipeline import TrainingPipeline
from src.predict import predict
from src.utils import load_config


def test_inference_pipeline(tmp_path):
    """Run training pipeline end-to-end and ensure prediction pipeline works."""
    n_samples = 20
    # Create synthetic dataset with equal class distribution
    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    rng = np.random.default_rng(42)
    rng.shuffle(labels)
    df = pd.DataFrame({
        "transaction_id": np.arange(n_samples),
        "amount": rng.uniform(1, 100, size=n_samples),
        "merchant_type": rng.choice(["retail", "travel"], size=n_samples),
        "device_type": rng.choice(["mobile", "desktop"], size=n_samples),
        "label": labels,
    })

    data_path = Path("data/raw/fraud_detection.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            os.environ["MLFLOW_REGISTRY_URI"] = tracking_uri
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_registry_uri(tracking_uri)

            config = load_config()
            experiment_name = config["mlflow_config"]["experiment_name"]
            client = MlflowClient(tracking_uri=tracking_uri)
            if not client.get_experiment_by_name(experiment_name):
                client.create_experiment(
                    experiment_name, artifact_location=f"{tmpdir}/artifacts"
                )

            pipeline = TrainingPipeline(model_name="logistic_regression", tune=False)
            pipeline.run()

            registered_model = (
                f"{config['mlflow_config']['registered_model_base_name']}-logistic_regression"
            )
            model_version = client.get_latest_versions(
                registered_model, stages=["Staging"]
            )[0]

            predict_input = tmp_path / "predict_input.csv"
            df.drop(columns=["label"]).to_csv(predict_input, index=False)
            predict_output = tmp_path / "predict_output.csv"
            predict(
                model_name="logistic_regression",
                stage="Staging",
                input_path=str(predict_input),
                output_path=str(predict_output),
            )

            result = pd.read_csv(predict_output)
            assert result.shape == (n_samples, 5)
            assert list(result.columns) == [
                "transaction_id",
                "amount",
                "merchant_type",
                "device_type",
                "prediction",
            ]
    finally:
        if data_path.exists():
            data_path.unlink()
        mlruns_path = Path("mlruns")
        if mlruns_path.exists():
            shutil.rmtree(mlruns_path)
