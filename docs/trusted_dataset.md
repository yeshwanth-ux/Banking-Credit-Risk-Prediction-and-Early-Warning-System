# Trusted analytical datasets

Day 5 provides two analytical views because the available data has two distinct
time models.

## Operational point-in-time dataset

`trusted_loan_analytical_dataset` has exactly one row for each unique
`(loan_id, prediction_date)` in `loan_prediction_dates`. It joins the customer
and loan records and derives repayment features only from installments due on or
before the prediction date.

Payments with a `payment_date` after the prediction date contribute zero paid
amount at that snapshot. Installments due after the prediction date are excluded.
Days past due are recalculated from dates rather than using the mutable stored
`days_past_due` field. Current `loan_status` and `repayment_status` are excluded
because they may have been updated after the historical prediction date.

Application-date snapshots can be inserted idempotently with:

```powershell
.\.venv\Scripts\python.exe -m src.data_processing --seed-application-dates
```

Scheduled early-warning dates should be inserted into `loan_prediction_dates`
by the future orchestration workflow. Dates before the loan application are
excluded from the trusted view and reported by data-quality checks.

## Research prototype dataset

`trusted_credit_risk_dataset` contains the validated South German Credit rows.
It removes the duplicate inverse source target `credit_risk` and retains only
`bad_credit` as the modeling label. The source has no loan identifiers, event
dates, prediction dates, or repayment events, so its rows cannot be represented
honestly as the operational point-in-time grain.

## Quality rules

The report command tests:

- unique analytical keys;
- required identifiers and values;
- valid numeric and label ranges;
- loan and customer key integrity; and
- prediction dates that do not precede applications.

It writes `reports/data_quality_weekly.md` and exits with status 1 if a check
fails. The report includes operational coverage, prototype coverage and class
balance, individual check results, and the interpretation limits of both views.

Apply migrations before generating the report:

```powershell
.\.venv\Scripts\python.exe -m src.migrations up
.\.venv\Scripts\python.exe -m src.data_processing
```
