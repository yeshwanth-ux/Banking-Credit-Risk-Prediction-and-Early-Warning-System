# Apache Airflow pipeline automation

Day 14 provides a local-development Airflow 3.3.2 environment and a daily
TaskFlow DAG named `credit_risk_daily_pipeline`. The Docker image is pinned and
installs the project dependencies separately from the application's virtual
environment.

The Compose configuration uses Airflow's standalone mode and SequentialExecutor
for local development. It is not a production deployment. The project directory
and task logs are mounted into the container so code, generated artifacts, and
execution logs remain inspectable on the host.

## Workflow

The DAG runs daily at 02:00 UTC with catchup disabled and one active run:

```text
validate_source
    -> ingest_source
    -> clean_and_validate
    -> generate_features
    -> batch_risk_prediction
    -> persist_predictions
    -> refresh_portfolio_reports
```

Every task receives two retries with exponential backoff, a five-minute initial
retry delay, a thirty-minute maximum delay, and a forty-five-minute execution
timeout. Task functions log input counts, output paths, model versions, scoring
run references, and report totals through Python logging; Airflow stores the
execution logs under `airflow/logs/`.

The task functions live in `src/pipeline.py` and do not import Airflow. This
keeps validation, prediction, persistence, and report refresh testable without a
running scheduler.

## Idempotency

Migration `V006__add_scoring_run_idempotency.sql` adds a unique idempotency key
to scoring runs. The DAG derives the key from its logical date and the SHA-256
of the input file. A retry or repeat execution for the same logical date and
source resolves the existing scoring run and prediction identifiers instead of
inserting another result set.

Raw prototype ingestion already uses deterministic source-record identifiers
and PostgreSQL upserts, so repeated ingestion updates those records rather than
duplicating them. Feature and prediction artifacts are written atomically into
a deterministic run directory.

## Local startup

Prerequisites are Docker Desktop with Docker Compose v2 and the configured
project PostgreSQL database. Copy `.env.example` to the ignored `.env` file and
set the real PostgreSQL password. On Windows and macOS, the default
`host.docker.internal` value connects the Airflow container to PostgreSQL on the
host.

Apply the project migration, build the image, and start Airflow:

```powershell
.\.venv\Scripts\python.exe -m src.migrations up
docker compose build
docker compose up
```

On its first startup, the container creates a container-local MLflow SQLite
registry from the frozen Day 10 model. This gives the Linux worker valid model
artifact paths while retaining the `CreditRiskEarlyWarning@champion` loading
contract. Airflow then initializes its own metadata database in the named
`airflow-state` volume.

Open `http://localhost:8080`. The Simple Auth Manager username is `admin`; read
its generated local-development password with:

```powershell
docker compose exec airflow cat /opt/airflow/state/simple_auth_manager_passwords.json.generated
```

Confirm the DAG parses without import errors:

```powershell
docker compose exec airflow airflow dags list
docker compose exec airflow airflow dags list-import-errors
```

Trigger the DAG from the UI, or execute a dated test run:

```powershell
docker compose exec airflow airflow dags test credit_risk_daily_pipeline 2026-10-28
```

Set `CREDIT_RISK_BATCH_LIMIT=10` in `.env` to restrict prediction to a small
development batch. An empty value scores every available prototype feature
record.

Stop services without deleting state:

```powershell
docker compose down
```

To rebuild the local Airflow and MLflow state from scratch, remove the Compose
volume explicitly and start again:

```powershell
docker compose down --volumes
docker compose up --build
```

## Verification

The automated tests use a two-record feature file and a deterministic model to
exercise batch scoring, idempotent persistence, and portfolio report refresh.
They also parse the DAG with a lightweight TaskFlow test harness and verify all
seven tasks, dependencies, schedule, retry count, exponential backoff, and
single-active-run policy. PostgreSQL verification repeats the same idempotency
key and confirms that only one prediction row is stored.
