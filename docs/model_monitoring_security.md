# Model monitoring and application security

Day 18 adds privacy-safe operational monitoring, input-distribution checks, risk
distribution summaries, scoring freshness, structured application events, and
role-aware access to reports. These controls support investigation; they do not
replace production observability, independent model validation, or security
operations.

## Structured logs and operational metrics

Every HTTP response produces a JSON log event containing only operational
metadata: UTC timestamp, service, event, generated request ID, HTTP method,
route path, status, latency, and active model version. Prediction-completion
events contain request type, result count, aggregate risk-band counts, and model
version. Failure events contain only the failure class and operation name.

Logs intentionally exclude request and response bodies, query values, API keys,
loan or customer references, feature names and values, probabilities, warning
evidence, personal information, and financial information. Internal exception
messages and tracebacks are not emitted by application failure events. Access
to container and platform logs must still be restricted and audited.

`GET /monitoring/summary` returns:

- total request and error counts, error rate, average latency, and counts by
  route;
- aggregate High, Medium, and Low prediction counts and proportions;
- the active registered model version;
- the latest scoring timestamp and its age in seconds;
- drift-window size and per-feature drift method, score, sample count, and
  status.

Metrics are in-memory and reset when the API process restarts. A production
deployment should export equivalent counters and histograms to a durable,
access-controlled monitoring system.

## Input drift

`config/training_distributions.json` contains aggregate distributions derived
only from the training split. It contains no source rows or identifiers.
Numerical features use ten training-quantile bins and Population Stability
Index (PSI). Categorical features use total variation distance (TVD). The API
keeps at most 1,000 recent validated feature rows in process memory and reports
`insufficient_data` until at least 30 observations exist.

The initial alert threshold is 0.20 for both PSI and TVD. An alert is a prompt
to investigate data contracts, population coverage, missingness, pipeline
changes, and category mappings. Drift alone does not prove that predictive
performance deteriorated and must not automatically trigger model replacement.
Regenerate the aggregate profile only from an approved training split:

```powershell
.\.venv\Scripts\python.exe -m src.monitoring
```

## Authentication and roles

API keys and their roles are configured server-side through the ignored
`RISK_ROLE_API_KEYS` environment value. It is a JSON object whose keys are the
secrets and whose values are role lists. Never commit real keys. Rotate keys
through the deployment secret manager and use TLS outside an isolated
workstation.

| Role | Aggregate portfolio | Account risk records | Monitoring summary |
|---|---:|---:|---:|
| `risk_viewer` | yes | no | no |
| `risk_analyst` | yes | yes | no |
| `monitor` | no | no | yes |
| `admin` | yes | yes | yes |

The legacy `RISK_REPORT_API_KEY` remains a local-development fallback and is
granted viewer, analyst, and monitor access. Role-aware deployments should use
`RISK_ROLE_API_KEYS`. Authentication failures return generic messages and do
not reveal whether a particular key or role exists.

## Review and retraining triggers

Manual review should begin when any of the following occurs:

- a drift alert persists across two complete, representative scoring windows;
- the High Risk proportion changes materially from its approved operating
  range or changes abruptly without a documented portfolio explanation;
- the latest scoring result is older than the expected batch schedule plus its
  agreed recovery window (24 hours is the initial daily-pipeline escalation);
- API error rate exceeds 2% over a representative 15-minute window, p95 latency
  breaches the service objective, or persistence failures occur;
- a model or configuration version changes unexpectedly;
- source schema, category coding, eligibility rules, target definition, or
  repayment policy changes;
- delayed outcomes show material calibration, recall, precision, fairness, or
  stability deterioration.

Retraining should be considered only after confirming data quality and having
enough newly observed outcomes for the configured prediction horizon. Candidate
models must repeat leakage review, time/customer-aware validation, calibration,
threshold analysis, fairness assessment, independent review, registry approval,
and controlled deployment. A drift alert by itself is not authorization to
retrain or promote a model.

Immediate manual escalation is appropriate for unexplained High Risk surges,
stale scoring, missing source feeds, authentication anomalies, repeated
prediction failures, or evidence that sensitive data appeared in logs. Customer
impacting decisions must remain subject to approved human and governance review.
