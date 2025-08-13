# Migration Guide: MLflow 2.10.2 to 2.22.1

This document records the updates required to bring the project from **MLflow 2.10.2** to **MLflow 2.22.1**. The upgrade aligns the codebase with the most recent 2.x release and patches known vulnerabilities.

## 1. Summary of Changes

The upgrade focused on the following areas:

1. **Dependency & Environment Alignment**
   - `pyproject.toml` now pins `mlflow` to `2.22.1` along with newer, compatible versions of `scikit-learn` (`1.7.1`) and `imbalanced-learn` (`0.13.0`).
   - `docker-compose.yml` installs `mlflow[auth]==2.22.1` so the tracking server and client run the same version.
2. **Model Registration Behavior**
   - In MLflow 2.22.1 the `ModelInfo` object returned by `mlflow.sklearn.log_model` once again exposes `registered_model_version`.
   - The project previously used `MlflowClient().get_latest_versions()` as a workaround in 2.10.2; this call is now **deprecated** and should be replaced with the new `registered_model_version` attribute when convenient.

## 2. API Notes

- `mlflow.set_registry_uri()` is still available but an equivalent function exists at `mlflow.config.set_registry_uri()`. Prefer the latter for forward compatibility.
- `MlflowClient().get_latest_versions()` is deprecated (model registry stages will be removed in a future release); rely on `ModelInfo.registered_model_version` instead.
- No breaking API changes were encountered when upgrading from 2.10.2 to 2.22.1.

## 3. Operational Runbook

1. **Set up Environment Variables**
   Copy the `.env.example` file to `.env` and adjust credentials if needed:
   ```bash
   cp .env.example .env
   ```

2. **Launch Services**
   Start the MLflow tracking server and MinIO artifact store:
   ```bash
   docker-compose up -d
   ```
   - MLflow UI: `http://localhost:5000`
   - MinIO Console: `http://localhost:9001`

3. **Install Dependencies**
   ```bash
   poetry install
   ```

4. **Run Tests**
   ```bash
   poetry run pytest
   ```

5. **Run a Training Pipeline**
   Use the Docker-based workflow or run locally inside the Poetry environment as described in `README.md`.

## 4. Security Hardening Notes

- Keep the MLflow server behind a private network and use strong credentials.
- Manage secrets outside of version control and rotate them regularly.
- Maintain pinned dependency versions and monitor for CVEs.
- Restrict permissions on the artifact store to the minimum required.

## Historical Context

Earlier versions of this repository documented a downgrade from MLflow 3.x to 2.10.2. Those instructions are retained in Git history for reference but are no longer relevant to the current setup.
