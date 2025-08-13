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
- Python 3.11+
  - If needed, install it using a version manager such as [pyenv](https://github.com/pyenv/pyenv):
    ```bash
    pyenv install 3.11
    pyenv global 3.11
    ```
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

## How to Use the Pipeline

### 1. Run Model Training (via Docker)

The training pipeline is designed to be run inside its Docker container to ensure a consistent environment.

1.  **Build the Docker image:**
    ```bash
    docker build -t fraud-detection-pipeline .
    ```

2.  **Run the training pipeline:**
    The container's `entrypoint.sh` script will automatically pull data from DVC before running the training script.
    ```bash
    # You must have a running MLflow server (see Docker Compose setup)
    # The default MLFLOW_TRACKING_URI is http://mlflow:5000
    # To connect to a server running on the host from the container, use --network="host"

    docker run --rm --network="host" \
      -e MLFLOW_TRACKING_URI="http://localhost:5000" \
      -e MLFLOW_S3_ENDPOINT_URL="http://localhost:9000" \
      -e AWS_ACCESS_KEY_ID=$(grep MINIO_ACCESS_KEY_ID .env | cut -d '=' -f2) \
      -e AWS_SECRET_ACCESS_KEY=$(grep MINIO_SECRET_KEY .env | cut -d '=' -f2) \
      fraud-detection-pipeline \
      --model xgboost --tune
    ```

    You can select different run modes for the training script:

    - `--tune`: run hyperparameter search and register the best model.
    - `--tune-and-evaluate`: first perform hyperparameter tuning, then immediately run cross-validation with the best parameters. Use this when you want a single command that both discovers optimal hyperparameters and reports evaluation metrics without updating `params.yaml`.
    - `--use-best-params`: skip tuning and run cross-validation using the `best_params` stored in `params.yaml`.

    Example using `--tune-and-evaluate`:

    ```bash
    docker run --rm --network="host" \
      -e MLFLOW_TRACKING_URI="http://localhost:5000" \
      -e MLFLOW_S3_ENDPOINT_URL="http://localhost:9000" \
      -e AWS_ACCESS_KEY_ID=$(grep MINIO_ACCESS_KEY_ID .env | cut -d '=' -f2) \
      -e AWS_SECRET_ACCESS_KEY=$(grep MINIO_SECRET_KEY .env | cut -d '=' -f2) \
      fraud-detection-pipeline \
      --model xgboost --tune-and-evaluate
    ```

### 2. Make Predictions with a Trained Model

Use `src/predict.py` to make predictions with a model from the MLflow Model Registry.
```bash
# Example: Make predictions with the 'champion' alias of the xgboost model
poetry run python src/predict.py \
  --model xgboost \
  --alias champion \
  --input "path/to/your/data.csv" \
  --output "predictions.csv"
```

### 3. Monitor for Data Drift

You can check for data drift between a reference dataset (e.g., the training data) and a new dataset.
```bash
poetry run python scripts/monitor_drift.py \
  --reference_data "data/raw/fraud_detection.csv" \
  --current_data "path/to/new/data.csv"
```
This will output a report indicating whether significant drift was detected for any features.
