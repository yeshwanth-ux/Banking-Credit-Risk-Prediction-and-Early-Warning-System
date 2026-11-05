# Banking Credit Risk Prediction and Early-Warning System

An end-to-end portfolio project for estimating loan default risk, identifying
early-warning signals, and producing explainable risk information for banking
teams.

## Project objectives

- Ingest and validate customer, loan, repayment, and account data.
- Store trusted records and model outputs in PostgreSQL.
- Engineer financial and repayment-behavior risk indicators.
- Train and evaluate interpretable credit-risk models.
- Assign configurable Low, Medium, and High risk bands.
- Serve predictions through FastAPI and automate batch workflows with Airflow.
- Track model experiments with MLflow and generate portfolio risk reports.

## Prediction targets

The intended operational target is `default_within_90_days`, a binary label
indicating whether a loan enters default during the 90 days after a prediction
timestamp. Only information available at or before that timestamp may be used.
The public research model instead predicts South German Credit `bad_credit`
(contract non-compliance), so its probability is a prototype proxy—not a
validated 90-day probability of default. The definitions and leakage controls
are documented in [docs/target_definition.md](docs/target_definition.md).

## Technology stack

- Python 3.11+
- Pandas, NumPy, and scikit-learn
- PostgreSQL and SQLAlchemy
- FastAPI and Pydantic
- MLflow
- Apache Airflow, run through Docker during the orchestration phase
- Pytest
- Docker and Docker Compose

## Repository structure

```text
.
|-- airflow/             # Airflow DAGs
|-- api/                 # FastAPI application
|-- data/
|   |-- raw/             # Immutable source data
|   `-- processed/       # Validated and model-ready data
|-- docs/                # Architecture and project documentation
|-- models/              # Serialized model artifacts
|-- notebooks/           # Exploration and analysis notebooks
|-- reports/             # Generated risk reports
|-- sql/                 # PostgreSQL schema and queries
|-- src/                 # Data, feature, training, evaluation, and prediction code
|-- tests/               # Automated tests
|-- .env.example         # Safe configuration template
|-- Dockerfile           # Non-root FastAPI/MLflow runtime image
|-- pyproject.toml       # Project and development-tool configuration
`-- requirements.txt     # Python dependencies
```

## Local setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS and Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Update the local `.env` values before connecting to PostgreSQL. Never commit
passwords, API keys, customer data, or other secrets.

## Architecture

The system design, component boundaries, and end-to-end data flow are described
in [docs/architecture.md](docs/architecture.md).

The PostgreSQL tables, relationships, migration workflow, and local integration
test are documented in [docs/database.md](docs/database.md).

The point-in-time analytical views, leakage controls, data-quality rules, and
weekly quality report are documented in
[docs/trusted_dataset.md](docs/trusted_dataset.md).

Reusable financial, repayment, customer-history, and account-activity features
are documented in [docs/feature_engineering.md](docs/feature_engineering.md).

Historical pattern analysis, split strategy, and training-only preprocessing are
documented in [docs/data_preparation.md](docs/data_preparation.md).

Logistic Regression and Random Forest baseline training, validation metrics, and
the false-alert versus missed-default trade-off are documented in
[docs/baseline_models.md](docs/baseline_models.md).

Cross-validated model optimization, probability calibration, permutation
importance, and local feature contributions are documented in
[docs/model_optimization.md](docs/model_optimization.md).

Frozen risk bands, validation cost assumptions, historical warning rules, and
structured human-readable warning results are documented in
[docs/early_warning.md](docs/early_warning.md).

Local MLflow experiment tracking, model registration, artifact contents, and
prediction-service loading are documented in
[docs/mlflow_tracking.md](docs/mlflow_tracking.md).

FastAPI startup, health checks, single and batch prediction contracts, error
behavior, and OpenAPI documentation are described in
[docs/prediction_api.md](docs/prediction_api.md).

Persisted scoring runs, protected risk-record retrieval, portfolio aggregation,
and report filters are documented in
[docs/risk_reporting.md](docs/risk_reporting.md).

The daily Apache Airflow ingestion, validation, feature, scoring, persistence,
and report-refresh workflow is documented in
[docs/airflow_pipeline.md](docs/airflow_pipeline.md).

Pandas report exports, observed-period trend safeguards, and the required
analyst verification workflow are documented in
[docs/risk_report_interpretation.md](docs/risk_report_interpretation.md).

Unit, API, PostgreSQL integration, Airflow validation, and complete-suite test
commands are documented in [docs/testing.md](docs/testing.md).

The Docker Compose stack, persistent volumes, health checks, environment
configuration, and prediction smoke test are documented in
[docs/containerization.md](docs/containerization.md).

Privacy-safe operational metrics, drift monitoring, scoring freshness,
role-aware report access, and review or retraining triggers are documented in
[docs/model_monitoring_security.md](docs/model_monitoring_security.md).

The executable raw-to-report workflow, verified Day 19 evidence, and failure
checks are documented in [docs/end_to_end_runbook.md](docs/end_to_end_runbook.md).
Dataset assumptions, exact validation metrics, and known limitations are
recorded in
[docs/project_assumptions_limitations.md](docs/project_assumptions_limitations.md).

The complete release documentation is indexed in
[docs/README.md](docs/README.md), with final diagrams in
[docs/release_architecture.md](docs/release_architecture.md) and release notes in
[RELEASE_NOTES.md](RELEASE_NOTES.md).

## Selected research dataset

Day 2 uses the UCI South German Credit dataset for reproducible exploratory
analysis. See [docs/dataset_card.md](docs/dataset_card.md) for source, license,
limitations, and permitted use, and [docs/data_dictionary.md](docs/data_dictionary.md)
for field definitions and leakage notes.

Download the ignored raw data from the official source:

```powershell
.\.venv\Scripts\python.exe -m src.download_data
```

## End-to-end verification

The Day 19 local run validated all 1,000 raw rows, generated 1,000 prototype
feature rows, persisted an idempotent 10-record scoring batch, retrieved all 10
latest predictions through authenticated FastAPI endpoints, rejected invalid
API input with HTTP 422, and reconciled the final CSV and HTML reports. Run it
again with:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_end_to_end --max-records 10
```

Docker was unavailable on the verification host; Docker Compose configuration
and image contracts are tested, but real container startup remains an explicit
environment-specific verification step.

## Final model evaluation and demo

The frozen champion was evaluated once on the 200-row untouched test split at
the validation-selected 0.44 threshold: ROC-AUC 0.8143, PR-AUC 0.6594,
precision 0.4842, recall 0.7667, F1 0.5935, Brier score 0.1739, and expected
calibration error 0.1418. The confusion matrix is TN 91, FP 49, FN 14, TP 46.
See [the final evaluation](reports/final_test_evaluation.md) for interpretation.

Run the anonymized demonstration:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_release_demo
```

Sample inputs and generated prediction/report artifacts are under `demo/`.

## Data and responsible use

Use only public or authorized datasets whose licenses permit this project.
Never commit personally identifiable information or confidential banking data.
This project is a portfolio demonstration and is not approved for production
lending decisions without governance, fairness testing, independent validation,
security review, and applicable regulatory approval.
