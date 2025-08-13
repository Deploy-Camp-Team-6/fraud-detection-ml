# Migration Guide: MLflow 3.x to 2.10.2

This document outlines the changes made to downgrade the project's MLflow integration from version 3.x to **2.10.2**. The primary motivation for this downgrade was to ensure stability and compatibility with a well-established, long-term support version of MLflow.

## 1. Summary of Changes

The core of the refactoring involved two main areas:
1.  **Dependency & Environment Alignment**: The MLflow version was inconsistent across the project. The `pyproject.toml` specified `mlflow^2.10.2`, but the `docker-compose.yml` for the tracking server was installing `mlflow==3.*`. This mismatch is a common source of subtle bugs.
2.  **API Incompatibility Fixes**: The application code was written against the MLflow 3.x API, which introduced breaking changes. A key `AttributeError` was discovered and fixed, related to how model version information is retrieved after logging a model to the registry.

### Key Changes Made:
-   **`pyproject.toml`**: Pinned `mlflow` to an exact version `2.10.2` to prevent automatic upgrades to incompatible minor versions. Also pinned `scikit-learn` and `imbalanced-learn` to compatible versions (`1.4.2` and `0.12.3` respectively) to resolve import errors.
-   **`docker-compose.yml`**: Updated the `mlflow-server` service to install `mlflow[auth]==2.10.2`, ensuring the client and server are running the same version.
-   **`src/pipeline/training_pipeline.py`**: Refactored the model registration logic. The `ModelInfo` object returned by `mlflow.sklearn.log_model` in MLflow 2.x does not contain the `registered_model_version` attribute. The code now uses `MlflowClient().get_latest_versions()` to fetch the newly created version information.
-   **`tests/mlflow_compat_test.py`**: Added a new test file to explicitly validate the end-to-end MLflow 2.10.2 workflow, from logging and registering to loading and prediction. This test now passes and will help prevent future regressions.

## 2. API Replacements

The main API difference encountered was in the return value of `mlflow.sklearn.log_model` when registering a model.

| MLflow 3.x Pattern (Original) | MLflow 2.10.2 Pattern (Refactored) | File(s) Changed |
| ----------------------------- | ---------------------------------- | --------------- |
| `version = model_info.registered_model_version` | `latest = client.get_latest_versions(...)`<br>`version = latest[0].version` | `src/pipeline/training_pipeline.py`<br>`tests/mlflow_compat_test.py` |

**Note on `mlflow.set_registry_uri()`**:
The codebase uses `mlflow.set_registry_uri()`. This function is compatible with MLflow 2.10.2. In newer versions of MLflow, this function has been moved to `mlflow.config.set_registry_uri()`. While no change was required for this refactoring, it is recommended to adopt the new pattern in future updates to stay aligned with the latest API.

## 3. Operational Runbook

To run the local development and testing environment, follow these steps:

1.  **Set up Environment Variables**:
    Copy the `.env.example` file to a new file named `.env`:
    ```bash
    cp .env.example .env
    ```
    You can customize the credentials in this file if needed. These variables are used by Docker Compose to configure the MLflow and MinIO services.

2.  **Launch Services**:
    Start the MLflow tracking server and the MinIO artifact store using Docker Compose:
    ```bash
    docker-compose up -d
    ```
    - MLflow UI will be available at: `http://localhost:5000`
    - MinIO Console will be available at: `http://localhost:9001`

3.  **Install Dependencies**:
    With the services running, install the Python dependencies using Poetry. This will create a virtual environment with all the packages pinned to the correct versions from `poetry.lock`.
    ```bash
    poetry install
    ```

4.  **Run the Test Suite**:
    To verify that the environment is set up correctly and the code is working, run the test suite:
    ```bash
    poetry run pytest
    ```
    All tests, including the new MLflow compatibility test, should pass.

5.  **Run a Training Pipeline**:
    To run a full training pipeline (as described in `README.md`), you can use the provided Docker-based workflow or run it locally within the poetry environment.

## 4. Security Hardening Notes

For a production deployment of MLflow, consider the following security best practices:

-   **Network Isolation**: The MLflow tracking server should not be exposed to the public internet. It should be placed within a private network (e.g., a VPC) and accessed only by authorized services and users.
-   **Authentication**: The current `docker-compose.yml` setup uses `mlflow[auth]`, which enables basic authentication. For a production scenario, ensure you are using strong, unique credentials for the tracking server, and consider integrating with a more robust authentication provider if your security requirements demand it. The credentials should be managed via a secrets management system (e.g., AWS Secrets Manager, HashiCorp Vault) rather than environment files.
-   **Artifact Store Security**: The artifact store (MinIO/S3) should have strict access policies. The credentials used by the MLflow server to access the artifact store should have the minimum required permissions (e.g., read, write, list, but not delete if possible, or have deletion protection enabled).
-   **Pinned Dependencies**: This refactoring effort pinned all major dependencies to specific versions (e.g., `mlflow=="2.10.2"`). This is a critical security practice to prevent a `pip install` from pulling in a newer version of a library that may contain a known vulnerability (CVE). Regularly scan your dependencies for known CVEs and have a process for safely upgrading them.
-   **User Permissions**: If you have multiple users or teams interacting with the same MLflow server, use MLflow's experiment-level permissions (if your backend store supports it) to control access and prevent accidental or malicious changes to other teams' work.
