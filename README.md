# Production-Grade Fraud Detection Pipeline

This project implements a complete, versioned, and reproducible MLOps pipeline for training a fraud detection model. It uses industry-standard tools like DVC for data versioning, MLflow for experiment tracking and model registry, and Docker for containerization.

## 🏛️ Architecture

* **Code Versioning**: Git
* **Data Versioning**: DVC with MinIO S3 as remote storage.
* **Experiment Tracking**: MLflow connected to a PostgreSQL backend.
* **Preprocessing**: Scikit-learn pipelines for robust feature transformation.
* **Modeling**: XGBoost for its performance on structured, imbalanced data.
* **Containerization**: Docker to ensure a consistent and reproducible runtime environment.

---

## 🚀 How to Run an Experiment

This project is designed for easy experimentation with different models. You can select the model to train via a command-line argument.

### Available Models
* `xgboost`
* `lightgbm`
* `logistic_regression`

### Running a Specific Model

To run the pipeline, you build the Docker image once and then pass the model name as an argument to the `docker run` command.

**1. Build the Docker Image (if you haven't already):**
```bash
docker build -t fraud-detection-pipeline .

# Connect to the docker network where mlflow/minio are running
docker run --rm --network=fraud_detection_project_default fraud-detection-pipeline --model lightgbm

docker run --rm --network=fraud_detection_project_default fraud-detection-pipeline --model xgboost

docker run --rm --network=fraud_detection_project_default fraud-detection-pipeline --model logistic_regression