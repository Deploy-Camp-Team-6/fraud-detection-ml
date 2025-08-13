from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def test_mlflow_version():
    """Asserts that the installed MLflow version is 2.10.2."""
    assert mlflow.__version__ == "2.10.2"

@pytest.fixture(scope="module")
def mlflow_client(tmpdir_factory):
    """
    A fixture to set up a temporary MLflow tracking server for the test module.
    """
    tmp_path = tmpdir_factory.mktemp("mlflow")
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    # In 2.10.2, it's good practice to set both, even if they are the same
    mlflow.set_registry_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

def test_mlflow_end_to_end_roundtrip(mlflow_client):
    """
    Tests a full MLflow round-trip: logging, registering, loading, and predicting.
    """
    experiment_name = "mlflow-compat-test"
    mlflow.set_experiment(experiment_name)

    # 1. Start a run and log items
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log parameters
        params = {"model_type": "logistic_regression", "solver": "lbfgs"}
        mlflow.log_params(params)

        # Log a metric
        mlflow.log_metric("some_metric", 0.95)

        # Log an artifact (a simple text file)
        artifact_path = "test_artifact.txt"
        with open(artifact_path, "w") as f:
            f.write("hello mlflow")
        mlflow.log_artifact(artifact_path, "text_files")
        Path(artifact_path).unlink()

        # 2. Train and log a model
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = LogisticRegression()
        model.fit(X, y)

        model_name = "test-compat-model"
        model_alias = "champion"

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=X[:5],
        )

        # 3. Set an alias on the registered model
        client = mlflow_client
        # In MLflow 2.x, log_model's returned ModelInfo does not contain the registered model version.
        # We fetch it manually from the registry.
        latest_version = client.get_latest_versions(name=model_name, stages=["None"])[0]
        client.set_registered_model_alias(
            name=model_name,
            alias=model_alias,
            version=latest_version.version
        )

    # 4. Load the model back using the alias
    model_uri = f"models:/{model_name}@{model_alias}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # 5. Perform a prediction and assert correctness
    X_test, _ = make_classification(n_samples=1, n_features=5, random_state=43)
    prediction = loaded_model.predict(X_test)

    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)

    # Verify that the run and model exist
    retrieved_run = client.get_run(run_id)
    assert retrieved_run.data.params == params
    assert "some_metric" in retrieved_run.data.metrics

    registered_model = client.get_registered_model(model_name)
    assert registered_model.name == model_name
    assert model_alias in registered_model.aliases
    assert registered_model.aliases[model_alias] == latest_version.version

    print("MLflow end-to-end compatibility test passed.")
