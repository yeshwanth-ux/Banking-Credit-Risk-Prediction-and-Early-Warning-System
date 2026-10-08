# Prediction Target Definition

## Target

`default_within_90_days` is a binary target defined for each eligible loan at a
specific `prediction_timestamp`:

- `1`: the loan first meets the documented default condition during the next
  90 calendar days.
- `0`: the loan remains observable and does not meet that condition during the
  same 90-day window.

The operational default condition must follow the selected dataset's documented
repayment outcome. When payment history supports it, the preferred condition is
90 or more days past due, charge-off, or an equivalent terminal default status.
Dataset-specific status mappings must be versioned with the data dictionary.

## Prediction point

The model scores an active, eligible loan at the end of an observation period.
All feature values must have an event time at or before `prediction_timestamp`.
The 90-day performance window begins immediately after that timestamp.

## Eligibility

A training row is eligible when:

1. the loan is active and has a valid customer and loan identifier;
2. required origination and prediction dates are known;
3. the observation period contains enough permitted history for the selected
   features; and
4. the complete 90-day outcome window is observable, unless the dataset provides
   a reliable terminal default event before the window ends.

Loans with incomplete outcome windows are censored and excluded from supervised
training rather than labeled as non-defaults.

## Leakage controls

The following information must not be used as a predictor when it occurs after
the prediction timestamp:

- default, charge-off, collection, or recovery outcomes;
- future payments, balances, delinquencies, or account statuses;
- manually assigned fields created after the outcome was known;
- aggregate statistics calculated using validation or test periods; and
- transformations fitted on the full dataset instead of training data only.

Feature queries must filter records by event time, and train/validation/test
splits should be time-based when reliable timestamps exist. Customer-level
grouping must prevent the same customer from leaking across splits when repeated
loans or observations would make that possible.

## Evaluation timing

Training and model selection use training and validation data. The untouched test
set is evaluated once after the model, feature set, and risk-band thresholds are
frozen. This keeps the reported final metrics independent of model selection.

## Assumptions requiring confirmation

The 90-day horizon and precise default-status mapping are initial project
decisions. They must be confirmed against the selected dataset, its observation
frequency, and the intended review process before model training begins.
