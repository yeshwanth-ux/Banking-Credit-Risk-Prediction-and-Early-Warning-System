# End-to-end workflow and verification

This runbook executes the same scheduler-independent task functions used by the
Airflow DAG, then verifies persisted predictions through FastAPI and reconciles
the final CSV and HTML reports. The workflow is idempotent for a logical date
and source-file hash.

## Prerequisites

1. Create `.env` from `.env.example` and configure a development PostgreSQL
   database. Never use a production or customer database for this exercise.
2. Install dependencies and apply all migrations.
3. Download the public source data and generate the Day 7-11 model artifacts,
   including the local MLflow registry and frozen classification configuration.

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m src.migrations up
.\.venv\Scripts\python.exe -m src.download_data
```

## Execute and verify

Run a bounded local batch. The default local registry is `mlflow.db`; pass
`--tracking-uri` only when intentionally testing another registry.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_end_to_end `
  --logical-date 2026-11-04 `
  --max-records 10
```

The harness performs, in order:

1. raw CSV schema and record validation;
2. idempotent PostgreSQL ingestion and rejected-row accounting;
3. trusted-view quality checks and dataset cleaning;
4. prototype and operational feature generation;
5. registered-model batch scoring through the Airflow task contract;
6. idempotent scoring-run and prediction persistence;
7. authenticated FastAPI health, risk-record, and portfolio retrieval;
8. an invalid-input API request that must return HTTP 422;
9. CSV and HTML report generation, existence checks, and count reconciliation;
10. an `end_to_end_manifest.json` containing non-sensitive verification evidence.

Runtime files are written below ignored `data/processed/airflow/` and
`reports/airflow/` directories. Repeating the same logical date and unchanged
source reuses the scoring run instead of duplicating predictions.

## Verified Day 19 execution

The local execution completed on November 4, 2026 with verification status
`passed`:

| Check | Result |
|---|---:|
| Raw rows read and accepted | 1,000 / 1,000 |
| Rejected raw rows | 0 |
| Prototype feature rows | 1,000 |
| Operational feature rows | 0 |
| Batch predictions persisted | 10 |
| FastAPI health | HTTP 200 |
| Risk records retrieved through FastAPI | 10 |
| Invalid API input | HTTP 422 |
| Final portfolio reconciliation | 10 accounts |
| Risk distribution | 5 Medium, 5 Low, 0 High |

Only one observed scoring period existed, so the trend output correctly states
`insufficient_observed_periods`; it must not be presented as a historical trend.
Zero operational rows are expected because the public prototype dataset does
not populate the separate customer, loan, repayment, and account-activity
tables used by the operational point-in-time view.

## Airflow and Docker verification

The executable DAG is `credit_risk_daily_pipeline`. Its topology test verifies
all seven tasks, dependency order, retries, schedule, and single-active-run
policy. In a Docker-enabled environment, validate and execute the real scheduler
with:

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose exec airflow airflow dags list-import-errors
docker compose exec airflow airflow dags test credit_risk_daily_pipeline 2026-11-04
.\.venv\Scripts\python.exe -m scripts.smoke_test_compose
```

Docker was not available on the Day 19 Windows host, so container startup was
not claimed as executed there. The scheduler-independent task chain, live local
PostgreSQL persistence, local MLflow model, FastAPI retrieval, and reports were
executed directly.

## Failure checks

Automated tests cover an unavailable database with bounded retries, invalid raw
schema, invalid and missing API fields, unavailable model registry, inference
failure, persistence failure responses, authorization failures, and report
reconciliation. Run everything, including PostgreSQL integration tests, with:

```powershell
$env:RUN_POSTGRES_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -q
```
