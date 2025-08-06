# Dockerfile

# --- Base Stage ---
FROM python:3.9-slim AS base
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app

# --- Builder Stage ---
FROM base AS builder
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
# Install dependencies, including dev for testing if needed later
RUN poetry install --no-root

# --- Final Stage ---
FROM base AS final
# Copy virtual env from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install MinIO client for setup script in docker-compose
RUN apt-get update && apt-get install -y curl && \
    curl "https://dl.min.io/client/mc/release/linux-amd64/mc" --create-dirs -o /usr/local/bin/mc && \
    chmod +x /usr/local/bin/mc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Default command to run the training pipeline
ENTRYPOINT ["python", "train.py"]
CMD ["--model", "xgboost"]