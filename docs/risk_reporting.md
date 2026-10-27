# Risk results and portfolio reporting

Day 13 stores every successful single or batch prediction as an immutable
scoring run. Migration `V005__add_scoring_runs_and_reporting.sql` creates the
`scoring_runs` table and extends `risk_predictions` with the external reference,
scoring-run relationship, MLflow model URI, and classification-configuration
version.

Run the migration before starting the API:

```powershell
.\.venv\Scripts\python.exe -m src.migrations up
```

Each prediction stores its UTC timestamp, registry name and version, model URI,
configuration version, probability, final risk band, warning reasons, and input
feature snapshot. The feature snapshot remains in PostgreSQL for controlled
audit use and is never returned by the reporting API.

## Report authentication

Set a long random value in the ignored `.env` file:

```dotenv
RISK_REPORT_API_KEY=replace-with-a-long-random-secret
```

Clients send it through the `X-API-Key` header. Missing or incorrect credentials
receive HTTP 403. If the server has no configured key, report routes return HTTP
503. Keys are compared using a constant-time comparison and are never stored in
scoring records or returned in responses.

## Permitted risk records

`GET /risk-records` returns a restricted projection of the latest prediction per
loan or external reference. It includes reference identifiers, loan status,
timestamp, model metadata, probability, risk band, and warning reasons. Customer
attributes and model feature snapshots are excluded.

Supported query parameters:

- `prediction_from` and `prediction_to`: ISO 8601 timestamps;
- `loan_status`: one of the statuses allowed by the loan schema;
- `risk_category`: `low`, `medium`, or `high`;
- `limit`: 1 to 500 records, default 100.

Example:

```powershell
$headers = @{ "X-API-Key" = $env:RISK_REPORT_API_KEY }
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/risk-records?loan_status=active&risk_category=high" `
  -Headers $headers
```

## Portfolio summary

`GET /reports/portfolio` accepts the same date, loan-status, and risk-category
filters. It returns the number and proportion of High, Medium, and Low accounts,
the total number of accounts, and the earliest and latest timestamps represented.

The SQL ranks results by prediction timestamp and identifier, retaining one
latest record per linked loan or external reference. The risk-category filter is
applied after ranking, preventing an older result from replacing a newer result
that belongs to another category. Counts therefore describe latest account
states rather than the number of scoring events.

Database integration tests insert repeated predictions inside a rollback-only
transaction and assert that the summary counts equal the records selected by the
same filters. This validates High, Medium, and Low counts without leaving test
data behind.
