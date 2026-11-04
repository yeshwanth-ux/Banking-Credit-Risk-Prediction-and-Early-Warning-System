# Project assumptions, limitations, and evaluation record

## Dataset and target assumptions

- The research prototype uses 1,000 corrected South German Credit contracts
  sampled from 1973-1975. Bad credits were deliberately oversampled to 30%.
- The modeled label is `bad_credit`, meaning contract non-compliance. It is not
  the operational target `default_within_90_days` and is not a validated
  probability of default.
- The public data contain no customer identifiers, event dates, direct income,
  payment transactions, balances, or delinquency history. Operational early-
  warning features and time-based validation cannot be demonstrated with it.
- Credit amounts are transformed Deutsche Mark values with undocumented
  original scaling. They preserve ordering but are not current currency values.
- Source codes are treated as categorical even where their labels are ordinal.
  Feature availability at a real decision timestamp cannot be proven for every
  field.
- The random stratified split assumes independent rows because neither dates nor
  stable customer groups exist. A real portfolio must use time-aware and
  customer-group-aware evaluation.

## Evaluation record

The reproducible split contains 600 training, 200 validation, and 200 held-out
test records, each with a 30% `bad_credit` rate. Preprocessing is fitted only on
training data. Day 9 model selection uses five-fold stratified cross-validation
and validation average precision; the held-out test split is explicitly not
used for selection or reported as a final unbiased performance estimate.

The selected logistic-regression prototype produced these validation results at
the evaluation threshold of 0.50:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7661 |
| PR-AUC | 0.6166 |
| Precision | 0.4574 |
| Recall | 0.7167 |
| F1 | 0.5584 |
| Brier score | 0.2061 |
| Expected calibration error | 0.1811 |
| False positives / false negatives | 51 / 17 |

These figures describe a 200-row validation sample and have material sampling
uncertainty. The operational risk thresholds (0.44 Medium and 0.84 High) were
selected from validation cost assumptions and are not the 0.50 evaluation
threshold. They require recalibration and governance for any new population.

## System limitations

- This is a research portfolio system, not a production lending platform or an
  approved automated decision system.
- No real customer data, production identity provider, TLS termination, key
  vault, audit platform, managed database, backup process, or disaster recovery
  is included.
- API metrics and drift windows are process-local and reset on restart.
- Drift detects input-distribution change; it does not measure outcome-based
  model degradation or establish causality.
- Fairness-sensitive fields are present. No claim of legal compliance, fairness,
  explainability sufficiency, or absence of proxy discrimination is made.
- The generated rule warnings are deterministic review signals. Public prototype
  data lack the repayment history needed to exercise real early-warning rules.
- One scoring period cannot support a trend conclusion. Simulated or example
  periods must never be described as observed history.
- Local MLflow, SequentialExecutor Airflow, SQLite service metadata, and Docker
  Compose are development choices and require production hardening.
- Docker startup could not be executed on the Day 19 host because Docker was not
  installed; container contract tests and documentation do not replace a real
  image-build and runtime smoke test.

Before real use, obtain representative authorized data with outcome maturity,
define the regulated use and operational target, repeat leakage and fairness
reviews, perform independent validation, recalibrate probabilities and
thresholds, implement human-review controls, complete security testing, and
obtain model-risk and legal approval.
