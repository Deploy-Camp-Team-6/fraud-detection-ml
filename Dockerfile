# Dockerfile

# --- Base Stage ---
# Use a slim Python base image for a smaller final image size.
FROM python:3.10-slim AS base

# Set environment variables to prevent Python from writing .pyc files and buffering output.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- Builder Stage ---
# This stage installs dependencies. It's separate to leverage Docker's layer caching.
# If dependencies in pyproject.toml haven't changed, this layer will be reused, speeding up builds.
FROM base AS builder

# Set the working directory in the container.
WORKDIR /app

# Install Poetry, the dependency manager.
RUN pip install poetry

# Copy only the dependency definition files.
COPY pyproject.toml poetry.lock ./

# Install project dependencies using Poetry.
# --no-root: Don't install the project package itself yet.
# --no-dev: Exclude development dependencies (e.g., testing libraries).
RUN poetry install --no-root --no-dev

# --- Final Stage ---
# This stage builds the final, runnable image.
FROM base AS final

# Create a non-root user to run the application
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin -c "Docker image user" appuser

# Set the working directory.
WORKDIR /app

# Copy the installed virtual environment from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Activate the virtual environment for subsequent RUN, CMD, and ENTRYPOINT instructions.
ENV PATH="/app/.venv/bin:$PATH"

# Copy the entire project source code into the working directory.
COPY . .

# Change ownership of the app directory to the new user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# [Optional] Load environment variables from a .env file if it exists.
# In a real CI/CD pipeline, these would be injected as secrets.
# Make sure your main script can read them. For this project, they are not needed by the training script
# as DVC is used for local data setup and mlflow/minio are expected to be on the same docker network.
# ENV MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
# ENV MINIO_SECRET_KEY=${MINIO_SECRET_KEY}

# Copy the entrypoint script and make it executable
COPY scripts/entrypoint.sh /app/scripts/entrypoint.sh
RUN chmod +x /app/scripts/entrypoint.sh

# Define the entrypoint for the container.
# This script will pull data and then run the training pipeline.
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Set a default model to run if no arguments are provided to the entrypoint.
CMD ["--model", "xgboost", "--tune"]