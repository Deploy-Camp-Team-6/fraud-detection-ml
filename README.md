# Production-Grade Fraud Detection ML Pipeline

This project provides a fully functional, end-to-end machine learning pipeline for fraud detection. It has been refactored from an initial boilerplate to a robust and reproducible system that aligns with MLOps best practices. The pipeline handles data preprocessing, model training, hyperparameter tuning, evaluation, and inference, all within a containerized environment.

## Key Features & Fixes Implemented

- **Dataset Alignment:** The entire pipeline has been aligned to work with the specified dataset (`transaction_id`, `amount`, `merchant_type`, `device_type`, `label`).
- **Advanced Preprocessing:** Includes a `log_transformation` for the skewed `amount` feature.
- **Class Imbalance Handling:** All models (`XGBoost`, `LightGBM`, `LogisticRegression`) now correctly handle class imbalance, which is critical for fraud detection.
- **Robust Evaluation:** Implements **5-fold Stratified Cross-Validation** for reliable performance metrics, instead of a simple train-test split.
- **Hyperparameter Tuning:** Integrated `GridSearchCV` to find the best model parameters, runnable with a `--tune` flag.
- **Full MLOps Stack:**
    - **DVC:** For data versioning (though not fully integrated in this fix).
    - **MLflow:** For experiment tracking, model logging, and a model registry.
    - **Poetry:** For deterministic dependency management.
    - **Docker:** For a fully containerized and reproducible environment.
- **Inference Ready:** Includes a `predict.py` script to load registered models and make predictions on new data.

## Project Structure

```
.
├── Dockerfile
├── README.md
├── analysis_plots/
├── config
│   └── config.yaml
├── data
│   └── raw
│       └── fraud_detection.csv
├── notebooks
│   └── run_eda.py
├── params.yaml
├── pyproject.toml
└── src
    ├── __init__.py
    ├── components
    │   ├── data_transformation.py
    │   └── model_trainer.py
    ├── pipeline
    │   └── training_pipeline.py
    ├── predict.py
    └── utils.py
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

2.  **Install Dependencies:**
    Use Poetry to install the dependencies defined in `pyproject.toml`.
    ```bash
    poetry install
    ```

3.  **Set up MLOps Services (MLflow & MinIO):**
    The `docker-compose.yml` file required for this is not provided in the boilerplate, but you would typically start MLflow and a MinIO S3 storage backend this way. Assuming you have a `docker-compose.yml` for these services:
    ```bash
    # This is a conceptual step.
    # docker-compose up -d mlflow-db minio mlflow-server
    ```
    For this project, ensure your MLflow tracking server is running and accessible. The default URI is `http://mlflow:5000`, which implies it's running in a Docker network.

## How to Use the Pipeline

### 1. Run Data Analysis (Optional)

To understand the dataset, you can run the EDA script. This will generate plots in the `analysis_plots/` directory.

```bash
poetry run python notebooks/run_eda.py
```

### 2. Run Model Training & Evaluation

The main training pipeline is executed via `src/pipeline/training_pipeline.py`.

**To run with 5-fold Cross-Validation:**

This command trains the specified model, evaluates it using 5-fold stratified cross-validation, logs the average metrics to MLflow, and then trains and registers a final model on the full dataset.

```bash
poetry run python src/pipeline/training_pipeline.py --model xgboost
# Or for other models:
# poetry run python src/pipeline/training_pipeline.py --model lightgbm
# poetry run python src/pipeline/training_pipeline.py --model logistic_regression
```

**To run with Hyperparameter Tuning:**

Add the `--tune` flag to perform `GridSearchCV`. This will find the best hyperparameters, log them, and register the best-performing model pipeline to MLflow.

```bash
poetry run python src/pipeline/training_pipeline.py --model xgboost --tune
```

### 3. Make Predictions with a Trained Model

Once a model is trained and registered (e.g., in the 'Staging' phase), you can use `src/predict.py` to make predictions on new data.

1.  **Create a sample CSV file** for prediction (e.g., `sample_data.csv`):
    ```csv
    transaction_id,amount,merchant_type,device_type
    1001,250.75,travel,desktop
    1002,50.00,groceries,mobile
    ```

2.  **Run the prediction script:**
    ```bash
    poetry run python src/predict.py \
      --model xgboost \
      --stage Staging \
      --input sample_data.csv \
      --output predictions.csv
    ```
    This will load the 'Staging' version of the `FraudDetector-xgboost` model, make predictions on `sample_data.csv`, and save the results to `predictions.csv`.

## Running with Docker

The provided `Dockerfile` allows you to build a container for the project.

1.  **Build the Docker image:**
    ```bash
    docker build -t fraud-detection-pipeline .
    ```

2.  **Run the training pipeline inside the container:**
    You need to ensure the container can connect to your MLflow server. If MLflow is running in a Docker network named `my_network`, you can do:
    ```bash
    docker run --rm --network=my_network \
      fraud-detection-pipeline \
      --model xgboost --tune
    ```
    The `ENTRYPOINT` is set to `python src/pipeline/training_pipeline.py`, so you only need to pass the arguments.
