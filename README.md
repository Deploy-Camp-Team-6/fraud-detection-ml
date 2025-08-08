# Production-Grade Fraud Detection ML Pipeline

This project provides a fully functional, end-to-end machine learning pipeline for fraud detection. It has been reviewed and enhanced to align with MLOps best practices, making it robust, reproducible, and secure for production deployment.

## Key Features & Enhancements

- **Full MLOps Stack:**
    - **DVC:** For data versioning.
    - **MLflow:** For experiment tracking, model logging, and a model registry.
    - **Poetry:** For deterministic dependency management.
    - **Docker & Docker Compose:** For a fully containerized and reproducible environment, including the MLOps stack.
- **Automated CI/CD Pipelines:**
    - **Continuous Integration:** On every push/PR, the pipeline automatically runs linters, security scans (`safety`), and unit tests.
    - **Release-Triggered Training:** Creating a GitHub release (e.g., `xgboost/v1.0`) automatically triggers a production-grade training pipeline that runs hyperparameter tuning and registers the best model to MLflow.
- **Production-Ready Features:**
    - **Container Security:** The training container runs as a **non-root user** to enhance security.
    - **Configuration Management:** All configurations are managed via YAML files (`config.yaml`, `params.yaml`), with sensitive values loaded from environment variables.
    - **Data Drift Monitoring:** Includes a script (`scripts/monitor_drift.py`) to detect concept drift between datasets.
- **Advanced ML Features:**
    - **Class Imbalance Handling:** All models correctly handle class imbalance.
    - **Robust Evaluation:** Implements **5-fold Stratified Cross-Validation** and `GridSearchCV` for hyperparameter tuning.
    - **Inference Ready:** Includes a `predict.py` script to make predictions with registered models.

## Project Structure
```
.
├── Dockerfile
├── docker-compose.yml
├── README.md
├── config
│   └── config.yaml
├── data
│   └── raw
│       └── fraud_detection.csv.dvc
├── notebooks
│   └── run_eda.py
├── params.yaml
├── pyproject.toml
├── scripts
│   ├── entrypoint.sh
│   └── monitor_drift.py
└── src
    ├── ...
```

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Poetry (`pip install poetry`)

### Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Set up MLOps Services (MLflow & MinIO):**
    This project now includes a `docker-compose.yml` file to easily set up the required services.

    a. **Create your environment file:**
       Copy the example environment file:
       ```bash
       cp .env.example .env
       ```
       You can customize the credentials in `.env` if needed.

    b. **Launch the services:**
       ```bash
       docker-compose up -d
       ```
       This will start the MLflow server (http://localhost:5000) and MinIO console (http://localhost:9001).

3.  **Install Local Dependencies:**
    Use Poetry to install the project's Python dependencies.
    ```bash
    poetry install
    ```

## Development and Deployment Workflow

This section outlines the complete workflow from local development to deploying a new model version.

### 1. Local Development Workflow

Follow these steps to set up your local environment for development.

**a. Setup**

1.  **Activate Virtual Environment:** Use Poetry's shell to manage dependencies and run commands within the project's environment.
    ```bash
    poetry shell
    ```
2.  **Run Linters and Formatters:** This project uses `ruff` for code quality. Before committing, always run:
    ```bash
    # Check for linting errors
    ruff check .
    # Format the code
    ruff format .
    ```
3.  **Run Unit Tests:** Ensure all tests pass before pushing changes.
    ```bash
    pytest
    ```

### 2. Running Experiments

You can run the training pipeline either locally (for development and debugging) or via Docker (for production-like runs).

**a. Start MLOps Services**

First, ensure the MLflow and MinIO services are running via Docker Compose:
```bash
# This command starts MLflow (http://localhost:5000) and MinIO (http://localhost:9001)
docker-compose up -d
```

**b. Pulling Data with DVC**

The training data is versioned with DVC. To use it locally, you must configure DVC to connect to your local MinIO instance and pull the data.

1.  **Configure DVC:**
    ```bash
    # Make sure your .env file is populated from .env.example
    # These commands configure DVC to use the credentials from your .env file
    export $(grep -v '^#' .env | xargs)
    dvc remote modify minio endpointurl "http://localhost:9000"
    dvc remote modify minio access_key_id "$MINIO_ACCESS_KEY_ID"
    dvc remote modify minio secret_access_key "$MINIO_SECRET_KEY"
    ```
2.  **Pull the data:**
    ```bash
    dvc pull -f
    ```

**c. Running the Pipeline Locally**

For faster iteration during development, you can run the pipeline directly on your machine.

1.  **Set Environment Variables:** The pipeline needs credentials to connect to MLflow and MinIO.
    ```bash
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    export AWS_ACCESS_KEY_ID=$(grep MINIO_ACCESS_KEY_ID .env | cut -d '=' -f2)
    export AWS_SECRET_ACCESS_KEY=$(grep MINIO_SECRET_KEY .env | cut -d '=' -f2)
    export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
    ```
2.  **Execute the script:**
    ```bash
    # Run training with hyperparameter tuning
    poetry run python src/pipeline/training_pipeline.py --model xgboost --tune

    # Run training with standard cross-validation
    poetry run python src/pipeline/training_pipeline.py --model lightgbm
    ```
    -   `--model`: Choose from `xgboost`, `lightgbm`, or `logistic_regression`.
    -   `--tune`: An optional flag to perform hyperparameter tuning. If omitted, the pipeline runs standard cross-validation.

**d. Running the Pipeline with Docker**

This is the recommended way to run the full pipeline, as it mirrors the production environment.

1.  **Build the image:**
    ```bash
    docker build -t fraud-detection-pipeline .
    ```
2.  **Run the container:**
    ```bash
    docker run --rm --network="host" \
      -e MLFLOW_TRACKING_URI="http://localhost:5000" \
      -e AWS_ACCESS_KEY_ID=$(grep MINIO_ACCESS_KEY_ID .env | cut -d '=' -f2) \
      -e AWS_SECRET_ACCESS_KEY=$(grep MINIO_SECRET_KEY .env | cut -d '=' -f2) \
      -e MLFLOW_S3_ENDPOINT_URL="http://localhost:9000" \
      fraud-detection-pipeline \
      --model xgboost --tune
    ```

### 3. Releasing a New Model

The project is configured for **Release-Triggered Training**. This means a new model is trained and registered automatically when you create a new release on GitHub.

**a. Continuous Integration (CI)**

On every push or PR to the `main` branch, a CI pipeline runs linters, security scans, and unit tests. All checks must pass before code can be merged.

**b. How to Trigger a Release**

1.  Navigate to the **Releases** page of your GitHub repository.
2.  Click **"Draft a new release"**.
3.  In the **"Tag"** field, enter a tag in the format `model-name/version`.
    -   **Example:** `xgboost/v1.2.0`
    -   The `model-name` **must** be one of the models defined in the pipeline (e.g., `xgboost`).
    -   The `version` should ideally follow semantic versioning.
4.  Provide a release title and description.
5.  Click **"Publish release"**.

**c. What Happens Next**

Publishing the release triggers a GitHub Actions workflow that:
1.  Builds the production Docker image.
2.  Pulls the latest version of the data using DVC.
3.  Runs the training pipeline with hyperparameter tuning (`--tune`).
4.  Logs all metrics and parameters to MLflow.
5.  Registers the best-performing model to the MLflow Model Registry, ready for deployment.

### 4. Other Utilities

**a. Making Predictions**

Use `src/predict.py` to make predictions with a model from the MLflow Model Registry.
```bash
# Example: Make predictions with the 'Staging' version of the xgboost model
poetry run python src/predict.py \
  --model xgboost \
  --stage Staging \
  --input "data/raw/fraud_detection.csv" \
  --output "predictions.csv"
```

**b. Monitoring for Data Drift**

Check for data drift between a reference dataset and a new dataset.
```bash
poetry run python scripts/monitor_drift.py \
  --reference_data "data/raw/fraud_detection.csv" \
  --current_data "path/to/new/data.csv"
```
This will output a report indicating whether significant drift was detected.
