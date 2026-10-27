# FastAPI prediction service

The Day 12 service loads the MLflow `CreditRiskEarlyWarning@champion` model at
application startup. The registered artifact contains the fitted preprocessing
pipeline and classifier. Registry metadata is checked against the frozen risk
configuration before the application accepts prediction requests.

The current model estimates the South German Credit `bad_credit` research
outcome. The response field `predicted_default_probability` follows the API
contract, while `probability_label` and `target_definition` explicitly identify
this value as a prototype proxy rather than a validated 90-day probability of
default.

## Run locally

Start the service from the repository root after creating the Day 11 local
MLflow registry and applying the current PostgreSQL migrations:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.migrations up
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The interactive OpenAPI interfaces are available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Endpoints

### `GET /health`

Returns HTTP 200 with the loaded model URI, registry version, and configuration
version. If startup cannot load or verify the registered model, it returns HTTP
503 without exposing internal exception details.

### `POST /predict`

Accepts one `loan_reference`, the 23 model `features`, and optional
`warning_inputs`. The response includes the probability, model-only band, final
risk band, model explanations, rule warnings, version metadata, scoring-run
reference, prediction identifier, and UTC prediction timestamp. Successful
results are stored atomically in PostgreSQL.

### `POST /predict/batch`

Accepts one to 100 single-loan requests and returns predictions in the same
order. Invalid fields, missing fields, unknown fields, and out-of-range values
receive FastAPI's structured HTTP 422 validation response. Loan references must
be unique within one batch. The complete batch is stored as one scoring run.

Example request:

```json
{
  "loan_reference": "LOAN-1001",
  "features": {
    "duration_months": 24,
    "credit_amount_dm": 4000.0,
    "age_years": 35,
    "credit_amount_per_month": 166.67,
    "checking_account_status": 2,
    "credit_history": 2,
    "purpose": 3,
    "savings": 2,
    "employment_duration": 3,
    "installment_rate": 3,
    "personal_status_sex": 3,
    "other_debtors": 1,
    "present_residence": 3,
    "property": 2,
    "other_installment_plans": 3,
    "housing": 2,
    "number_credits": 1,
    "job": 3,
    "people_liable": 2,
    "telephone": 2,
    "foreign_worker": 1,
    "young_borrower_flag": 0,
    "long_duration_flag": 0
  },
  "warning_inputs": {
    "missed_payment_count": 1,
    "maximum_days_past_due": 35,
    "debt_to_income_ratio": 0.52,
    "previous_debt_to_income_ratio": 0.39
  }
}
```

The request and response schemas are authoritative in the generated OpenAPI
document. Batch processing is bounded at 100 loans to keep synchronous requests
predictable; scheduled portfolio scoring belongs in the batch pipeline.
