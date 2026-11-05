# v1.0.0 release checklist

Completed on November 5, 2026.

## Model and demonstration

- [x] Frozen model checksum matched the frozen configuration.
- [x] Untouched 200-row test split evaluated exactly once.
- [x] Test-data and model SHA-256 hashes recorded.
- [x] ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, Brier score,
  expected calibration error, and calibration bins published.
- [x] Three fresh, synthetic ingestion and prediction examples validated.
- [x] Low, Medium, and High demonstration outcomes generated.
- [x] Missed-payment, worsening-delay, debt-burden, severe-delinquency, and
  previous-default warning examples generated.
- [x] Anonymized CSV, JSON, and HTML report artifacts verified.

## Quality and repository audit

- [x] Ruff passed for application, orchestration, scripts, and tests.
- [x] Complete suite passed: 101 tests, including PostgreSQL integration.
- [x] Repository-local Python, pytest, Ruff, mypy, and coverage caches removed.
- [x] `.env` confirmed ignored and untracked.
- [x] Recognizable private-key and access-token patterns found in tracked files: 0.
- [x] No raw customer data or real customer identifiers included.
- [x] Generated release outputs contain repository-relative paths only.
- [x] Git whitespace validation passed.

## Documentation and release

- [x] README and documentation index finalized.
- [x] Runtime, daily batch, and online review diagrams published as Mermaid.
- [x] API, Docker, monitoring, end-to-end, and analyst runbooks linked.
- [x] Dataset assumptions, model limitations, and final evaluation recorded.
- [x] Release notes prepared.
- [x] Local annotated `v1.0.0` Git tag planned for the verified release commit.

Docker was not installed on the release host. Docker Compose contracts are
covered by automated tests, but an actual image build and multi-container smoke
test remains an environment-specific post-release verification item. No Git
commit or tag is pushed automatically.
