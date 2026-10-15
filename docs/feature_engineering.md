# Credit risk feature engineering

Day 6 adds reusable Pandas and NumPy transformations for operational loan
snapshots and the South German Credit research prototype.

## Operational features

The operational feature source begins with
`trusted_loan_analytical_dataset`. Every history query is bounded by the row's
`prediction_date`. It provides:

- monthly income and an amortizing estimated monthly installment;
- loan-to-income, installment-burden, debt-to-income, and outstanding-debt ratios;
- payment completion, missed-payment count and ratio, average delay, and maximum delay;
- prior 90-day delinquency-derived default count and flag;
- customer loan count, total principal, outstanding debt, and multiple-loan flag; and
- lifetime and trailing-90-day account activity, net flow, latest available balance,
  and recent-activity flag.

Outstanding customer debt uses loan principal less payments recorded on or
before the prediction date. The monthly customer obligation is a principal-only
proxy because the schema does not store a contractual installment amount for
every loan. This definition is intentionally named in the source data.

Credit utilization requires both a revolving balance and credit limit. The
current schema has balances but no credit limit, so the database query emits a
missing limit. `credit_utilization_ratio` remains missing and
`credit_utilization_available` is zero. The reusable function calculates the
ratio when an authorized source supplies both values.

Current loan and repayment status fields are excluded because they can change
after a historical prediction date. Previous defaults are inferred only when an
installment was at least 90 days past due as of the prediction date.

## Research prototype features

The South German Credit dataset has no income, account balance, credit limit,
transaction dates, or repayment events. The prototype transformation therefore
adds only defensible fields: credit amount per duration month, a young-borrower
indicator, and a long-duration indicator. It does not manufacture operational
financial ratios.

## Generate feature files

After applying database migrations and loading trusted data, run:

```powershell
.\.venv\Scripts\python.exe -m src.feature_engineering
```

This writes ignored local artifacts to `data/processed/operational_features.csv`
and `data/processed/prototype_features.csv`. Source identifiers and labels remain
available for downstream splitting; training code must explicitly separate them
from the feature matrix.
