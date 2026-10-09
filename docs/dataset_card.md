# South German Credit Dataset

## Selection decision

Day 2 uses the corrected South German Credit dataset from the UCI Machine
Learning Repository. UCI describes 1,000 credit records with 20 predictor
variables, including 700 good and 300 bad credit outcomes. The records are a
stratified sample of German credits from 1973 through 1975, with bad credits
intentionally oversampled.

The corrected dataset was selected instead of the older Statlog German Credit
copy because UCI reports severe coding-information errors and missing background
documentation in that older version.

## Source and attribution

- Dataset: South German Credit
- Repository: UCI Machine Learning Repository
- UCI dataset ID: 573
- DOI: https://doi.org/10.24432/C5QG88
- Dataset page: https://archive.ics.uci.edu/dataset/573/south+german+credit
- Official archive: https://archive.ics.uci.edu/static/public/573/south%2Bgerman%2Bcredit%2Bupdate.zip
- Citation: South German Credit [Dataset]. (2020). UCI Machine Learning Repository.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

The license permits sharing and adaptation for any purpose when appropriate
credit is provided. Redistributed or derived artifacts must retain attribution
and indicate material changes.

## Dataset shape and outcome

- Rows: 1,000 credit contracts
- Source predictors: 20
- Source outcome: `credit_risk`, where `1` means the contract was complied with
  and `0` means it was not complied with
- Modeling label: `bad_credit = 1 - credit_risk`
- Declared missing values: none
- Observed class balance: 300 bad credits and 700 good credits

The downloader preserves the official archive, source data, code table, and R
reader in `data/raw/`. It also creates `banking_data.csv` with descriptive English
column names and the derived `bad_credit` target. Raw files are intentionally
ignored by Git and can be reproduced with:

```powershell
.\.venv\Scripts\python.exe -m src.download_data
```

## Information groups

### Customer information

Age, personal-status/sex grouping, employment duration, residence duration,
housing, job category, dependents, telephone registration, and foreign-worker
status.

### Loan details

Credit duration, transformed credit amount, purpose, installment-rate band,
other debtors or guarantors, and other installment plans.

### Income and affordability indicators

The dataset has no direct income value. Installment rate is an ordinal band based
on disposable income. Checking-account status and savings are coarse financial
capacity indicators.

### Credit and repayment indicators

Credit history describes compliance with previous or concurrent contracts, and
number of credits summarizes contracts held at the bank. There is no dated
payment history, balance history, days-past-due series, or transaction history.

## Target and leakage assessment

`bad_credit` is the sole modeling target. Both `bad_credit` and its source inverse,
`credit_risk`, must be excluded from the feature matrix.

No predictor is explicitly documented as being recorded after the final outcome.
However, `credit_history` covers previous or concurrent contracts, and the source
does not provide event timestamps. Its availability at a real prediction point
cannot be proven and must be treated as a temporal-leakage risk. The same
time-availability review applies to checking-account status, number of credits,
and other installment plans before production use.

## Limitations

1. The data are from 1973 through 1975 and use Deutsche Mark values, so economic,
   lending, communication, and social patterns are not representative of current
   banking populations.
2. Bad credits were oversampled. The 30% observed bad-credit share is not a
   population default rate and predicted probabilities require recalibration for
   any deployment population.
3. Credit amount underwent an undocumented monotonic transformation. Values
   preserve ordering but cannot safely be interpreted as original monetary
   amounts.
4. There are no dates, customer identifiers, repayment events, or account-history
   observations. Time-based splitting, customer-group splitting, delinquency
   features, and early-warning trend analysis are impossible.
5. The label records contract compliance, not default within 90 days. Results
   must be described as bad-credit classification rather than a validated
   probability of default.
6. Several fields are discretized ordinal codes, so distances between codes are
   not necessarily meaningful.
7. `personal_status_sex`, `age_years`, and `foreign_worker` are sensitive or
   fairness-relevant. They require governance and fairness analysis and should
   not be used automatically in lending decisions.
8. The dataset is small. Validation results will have material sampling
   uncertainty and cannot establish production readiness.

## Permitted use

This dataset is appropriate for education, reproducible portfolio analysis, and
method development with attribution. It is not sufficient for real lending
decisions, regulatory validation, fairness claims, or current portfolio-risk
estimation.
