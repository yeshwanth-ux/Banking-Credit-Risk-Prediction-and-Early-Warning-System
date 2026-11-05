# Release notes

## v1.0.0 — Final portfolio release

Release v1.0.0 provides a reproducible research implementation of a banking
credit-risk and early-warning workflow:

- validated CSV ingestion and PostgreSQL analytical views;
- reusable financial and repayment feature engineering;
- baseline and optimized scikit-learn pipelines;
- MLflow experiment tracking and champion registration;
- frozen Low, Medium, and High risk bands plus deterministic warning rules;
- authenticated FastAPI prediction, risk, portfolio, and monitoring endpoints;
- idempotent Airflow batch scoring and analyst CSV/HTML reports;
- Docker Compose definitions for PostgreSQL, MLflow, API, and Airflow;
- structured privacy-safe logs, drift checks, and operational metrics;
- 95+ unit, API, database, orchestration, security, and end-to-end tests;
- one-time held-out test evaluation and anonymized release demonstration.

The model predicts historical South German Credit contract non-compliance. It
is not a validated current probability of default and is not approved for real
lending decisions. See the documentation index and limitations record before
interpreting any result.
