# Unit and integration testing

Day 16 validates the credit-risk system at function, service, database, and
workflow boundaries. Tests use deterministic fixtures and synthetic reference
identifiers; they do not require or contain real customer information.

## Coverage map

| Requirement | Test coverage |
|---|---|
| Data cleaning and validation | `test_data_processing.py`, `test_ingestion.py` |
| Financial and repayment features | `test_feature_engineering.py` |
| Valid, invalid, and missing model inputs | `test_predict.py`, `test_api.py` |
| Low, Medium, and High classification | `test_predict.py` |
| FastAPI contracts and error handling | `test_api.py` |
| PostgreSQL loading and reporting | `test_database_integration.py` |
| Airflow task behavior and DAG topology | `test_airflow_pipeline.py` |
| Analyst CSV and HTML outputs | `test_reporting.py` |

The model inference tests use deterministic probability doubles. They verify
that eligible records reach the model and that missing, null, or invalid
records are rejected before model execution. Boundary tests cover the exact
Medium and High thresholds, while end-to-end classifier tests cover all three
risk bands.

The PostgreSQL tests execute inside rollback-only transactions. They validate
inserts, point-in-time leakage controls, feature-source queries, latest-score
portfolio aggregation, scoring history, and generated report reconciliation
without retaining test records.

The Airflow validation imports the DAG through a lightweight TaskFlow harness.
It checks the schedule, retry policy, single-active-run setting, task list, and
every dependency, while separate task tests exercise a small repeatable batch.

## Running tests

Run fast unit, API, report, and workflow tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run lint for application and test code:

```powershell
.\.venv\Scripts\python.exe -m ruff check src api airflow tests
```

To include PostgreSQL integration tests, configure the ignored `.env` file,
apply all migrations, and enable the opt-in marker for the process:

```powershell
.\.venv\Scripts\python.exe -m src.migrations up
$env:RUN_POSTGRES_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -q
```

Integration tests must run against a development database whose schema is
managed by this project. The tests roll back their own records, but migration
commands still change that database schema.
