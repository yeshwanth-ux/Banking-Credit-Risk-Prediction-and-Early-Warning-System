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

## Prediction target

The primary target is `default_within_90_days`, a binary label indicating
whether a loan enters default during the 90 days after a prediction timestamp.
Only information available at or before that timestamp may be used as a model
feature. The complete definition, eligibility rules, and leakage controls are
documented in [docs/target_definition.md](docs/target_definition.md).

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
|-- Dockerfile           # API image definition, implemented in the Docker phase
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

## Selected research dataset

Day 2 uses the UCI South German Credit dataset for reproducible exploratory
analysis. See [docs/dataset_card.md](docs/dataset_card.md) for source, license,
limitations, and permitted use, and [docs/data_dictionary.md](docs/data_dictionary.md)
for field definitions and leakage notes.

Download the ignored raw data from the official source:

```powershell
.\.venv\Scripts\python.exe -m src.download_data
```

## Development status

Day 1 establishes the project structure, dependency manifests, environment
template, target definition, and high-level architecture. Functional pipeline,
database, modeling, API, orchestration, and reporting components are developed
in later milestones.

## Data and responsible use

Use only public or authorized datasets whose licenses permit this project.
Never commit personally identifiable information or confidential banking data.
This project is a portfolio demonstration and is not approved for production
lending decisions without governance, fairness testing, independent validation,
security review, and applicable regulatory approval.
