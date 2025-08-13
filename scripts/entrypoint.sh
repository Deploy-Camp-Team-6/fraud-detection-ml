#!/bin/bash
set -e

# Ensure the MinIO endpoint is available to all subprocesses
export MLFLOW_S3_ENDPOINT_URL

# Configure DVC with credentials from environment variables
dvc remote modify minio endpointurl ${MLFLOW_S3_ENDPOINT_URL}
dvc remote modify minio access_key_id ${AWS_ACCESS_KEY_ID}
dvc remote modify minio secret_access_key ${AWS_SECRET_ACCESS_KEY}

# Pull the data using DVC
echo "Pulling data with DVC..."
dvc pull --force
echo "Data pulled successfully."

# Execute the main command (the training pipeline)
echo "Starting training pipeline..."
exec python src/pipeline/training_pipeline.py "$@"
