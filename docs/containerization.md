# Docker containerization

Day 17 provides a local Docker Compose stack for PostgreSQL, MLflow, the
FastAPI prediction service, and Apache Airflow. A one-shot `model-init` service
applies PostgreSQL migrations and registers the frozen model after PostgreSQL
and MLflow become healthy. The API and Airflow start only after that initializer
finishes successfully.

## Prerequisites

- Docker Desktop with Docker Compose v2
- The ignored Day 7-10 generated artifacts, including
  `models/frozen_credit_risk_model.joblib` and
  `data/processed/validation.csv`
- Raw pipeline data at `data/raw/banking_data.csv` if the Airflow DAG will run

Copy the example environment file and replace both placeholder secrets. Never
commit `.env`.

```powershell
Copy-Item .env.example .env
```

At minimum, set a strong `POSTGRES_PASSWORD` and `RISK_REPORT_API_KEY`.
Host-port variables may be changed when ports 5432, 5000, 8000, or 8080 are
already occupied. Container-to-container addresses remain `postgres:5432` and
`mlflow:5000`; do not replace them with `localhost`.

## Services and persistence

| Service | Host port | Purpose |
|---|---:|---|
| PostgreSQL | 5432 | Trusted data, scoring runs, and predictions |
| MLflow | 5000 | Experiment tracking and model registry |
| FastAPI | 8000 | Health and prediction endpoints |
| Airflow | 8080 | Scheduled ingestion and batch scoring |

Named volumes retain PostgreSQL data, MLflow metadata and artifacts, Airflow
state and logs, and generated report artifacts across normal restarts. MLflow
artifacts are mounted read-only in the API and Airflow containers. The raw data
directory is mounted read-only into Airflow.

The application image installs a runtime-only dependency set, copies the
required model and application artifacts, and runs as the unprivileged
`creditrisk` user. The Airflow image also copies project code instead of
depending on a writable source-code bind mount.

## Build and start

Validate the rendered configuration, build images, and start the stack:

```powershell
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
```

The health checks cover PostgreSQL readiness, the MLflow health endpoint, the
FastAPI model-aware health endpoint, and Airflow's monitor endpoint. Inspect a
failed initializer or service with:

```powershell
docker compose logs model-init
docker compose logs api
docker compose logs airflow
```

## Startup and prediction smoke test

After the services start, run the checked-in smoke test. It retries `/health`,
sends one schema-valid prediction to `/predict`, and verifies the probability,
risk band, and model-version fields.

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_compose.py
```

The request uses a synthetic reference and is persisted in the local
development database. It is not a validated banking prediction.

Useful browser endpoints are:

- FastAPI documentation: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`
- Airflow UI: `http://localhost:8080`

## Stop or reset

Stop containers while retaining data:

```powershell
docker compose down
```

Deleting volumes permanently removes the local PostgreSQL database, MLflow
registry and artifacts, Airflow state and logs, and generated reports. Use this
only when a full reset is intended:

```powershell
docker compose down --volumes
```

Production deployment requires external secret management, TLS, restricted
network exposure, durable managed databases and artifact storage, Airflow
executor hardening, backups, monitoring, and independent security review.
