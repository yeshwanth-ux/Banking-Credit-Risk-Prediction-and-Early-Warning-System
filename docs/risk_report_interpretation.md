# Risk reporting and analyst interpretation

Day 15 produces an analyst-facing portfolio snapshot and an investigation
queue from persisted risk predictions. The reports are decision-support
artifacts for a research prototype; they are not automatic lending decisions,
proof of default, or substitutes for authorized source records.

## Generated files

The final Airflow task writes four files into the run-specific report folder:

- `portfolio_risk_summary.csv` contains a complete High, Medium, and Low risk
  distribution, including zero-count bands, portfolio proportions, and average
  predicted default probabilities.
- `high_risk_accounts.csv` contains the latest High Risk accounts, their
  predicted default probabilities, model and configuration versions, and
  rule-based warning reasons. Access must be restricted because reference
  identifiers may be sensitive.
- `risk_trends.csv` summarizes the latest observed score for each account on
  each available scoring day. Its `data_status` states whether at least two
  observed periods exist.
- `portfolio_risk_report.html` presents the same information in a readable
  report with interpretation and verification reminders.

Files are replaced atomically so a reader does not observe a partially written
report. The portfolio snapshot keeps only the newest persisted prediction for
each reference identifier within the selected scope.

## Interpretation

`default_probability` estimates the configured `default_within_90_days` target
under the frozen prototype model. It is neither a certainty nor an estimate for
a different horizon. A risk band is a configured threshold classification.
High Risk means that the account should be reviewed; it does not itself prove
default or authorize an adverse action.

Warning reasons are historical rule triggers, such as missed payments or
worsening debt burden. They must be distinguished from model explanations:
rules identify a condition that fired, while model explanations describe how
inputs influenced a model score. When no rule fired, the report explicitly
states that the account entered the queue through the model threshold only.

Risk trends contain only persisted, observed scoring periods. A single period
is labelled insufficient for a historical trend. Multiple periods support
descriptive comparison only. Analysts must not infer causation or claim
portfolio deterioration until they verify consistent population coverage,
source freshness, eligibility rules, model version, configuration version, and
data quality. The reporting code never manufactures simulated periods and
never presents demonstration data as historical evidence.

## Required analyst verification

Before escalation or action, an authorized analyst should:

1. Confirm the report date range, portfolio scope, loan status, and account
   population against the reporting request.
2. Reconcile every investigated account with the authorized loan, repayment,
   and account systems; do not rely on a CSV alone.
3. Verify source freshness, missing values, duplicate handling, and that every
   feature was available at the prediction timestamp.
4. Confirm the model version, preprocessing pipeline, risk configuration, and
   probability calibration are approved for the stated use.
5. Review rule triggers separately from model explanations and record any
   source evidence used to confirm or dismiss a warning.
6. Check monitored fairness, protected-class controls, access authorization,
   and applicable governance before any customer-impacting decision.
7. Investigate changes in population, eligibility, data, and model versions
   before describing differences between periods as a genuine risk trend.

The exported report should be retained and distributed according to the
organization's security, audit, and records-management policies.
