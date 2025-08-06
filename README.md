# Production-Grade Fraud Detection Pipeline

This project provides a complete, versioned, and reproducible MLOps pipeline for training a fraud detection model. It has been refactored and implemented to work with a custom dataset and includes a full MLOps stack for experimentation and deployment readiness.

## 🏛️ Architecture

*   **Code Versioning**: Git
*   **Data Versioning**: DVC with MinIO S3 as remote storage.
*   **Experiment Tracking**: MLflow with a PostgreSQL backend.
*   **Containerization**: Docker and Docker Compose.
*   **Preprocessing**: Scikit-learn pipelines for robust feature transformation.
*   **Modeling**: XGBoost, LightGBM, and Logistic Regression with hyperparameter tuning.
*   **Dependency Management**: Poetry.

---

## 🚀 How to Run the Project

### Prerequisites

*   Docker and Docker Compose installed.
*   An environment that can run shell scripts.

### Step 1: Set Up the Environment

First, launch the MLOps stack (MLflow, MinIO, PostgreSQL) using Docker Compose.

```bash
docker-compose up --build -d
```

This command will:
1.  Build the Docker image for the training environment.
2.  Start the MLflow, MinIO, and PostgreSQL services in detached mode.
3.  The MLflow UI will be available at `http://localhost:5000`.
4.  The MinIO console will be available at `http://localhost:9001` (login with `minioadmin`/`minioadmin`).

### Step 2: Prepare and Version the Data

1.  **Place your dataset** at `data/raw/fraud_detection.csv`. The expected columns are `transaction_id`, `amount`, `merchant_type`, `device_type`, and `label`.

2.  **Version the data with DVC**. This project uses DVC to track the dataset.
    *   First, you need to configure DVC to use the MinIO container as remote storage.
        ```bash
        # Configure DVC remote storage
        dvc remote add -d minio s3://dvc-remote/fraud-data
        dvc remote modify minio endpointurl http://localhost:9000
        dvc remote modify minio access_key_id minioadmin
        dvc remote modify minio secret_key_id minioadmin
        ```
    *   Now, add your data file to DVC tracking and push it to the remote storage.
        ```bash
        dvc add data/raw/fraud_detection.csv
        dvc push
        ```
    *   Commit the changes to Git.
        ```bash
        git add data/raw/fraud_detection.csv.dvc .dvc/config
        git commit -m "Track and version raw dataset"
        ```

### Step 3: Run the Training Pipeline

You can run the training pipeline for any of the supported models using a `docker-compose run` command. This will execute the `train.py` script inside the containerized environment.

**Available Models**:
*   `xgboost`
*   `lightgbm`
*   `logistic_regression`

**Example Commands:**

```bash
# Run training for XGBoost
docker-compose run --rm app --model xgboost

# Run training for LightGBM
docker-compose run --rm app --model lightgbm

# Run training for Logistic Regression
docker-compose run --rm app --model logistic_regression
```

The pipeline will perform cross-validated training, hyperparameter tuning, and log all results (parameters, metrics, artifacts) to the MLflow server. The best model for each run will be registered in the MLflow Model Registry.

### Step 4: Make Predictions

A prediction script, `predict.py`, is provided to make predictions using a saved model.

**Note:** This script is designed to be run locally for demonstration. It loads the model and preprocessor from the `saved_models/` directory, which is created during the training run.

**Example Usage:**

```bash
python predict.py \
  --model_dir saved_models/ \
  --input_json '[{"amount": 150.75, "merchant_type": "online_retail", "device_type": "desktop"}]'
```

This will output the fraud prediction (0 or 1) and the probability score for the input transaction.