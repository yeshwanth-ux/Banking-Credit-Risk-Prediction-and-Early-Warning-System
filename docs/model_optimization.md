# Model optimization and explainability

Day 9 tunes Logistic Regression and Random Forest pipelines with
`RandomizedSearchCV`. Because the research dataset has no reliable dates or
customer identifier, training uses shuffled five-fold `StratifiedKFold` with a
fixed random seed. Each fold preserves the bad-credit rate. A future operational
dataset with repeated customers must replace this with grouped or time-aware
cross-validation.

Average precision is the search objective because bad credit is the minority
class. Preprocessing remains inside each model pipeline, so imputation, scaling,
and encoding are fitted independently within each training fold. Eight parameter
combinations are evaluated per model. The validation dataset is used only after
the searches finish, and the held-out test dataset remains unopened.

## Calibration

Validation calibration is summarized with Brier score and equal-width expected
calibration error. Lower values are better. A reliability chart compares mean
predicted probability with the observed bad-credit rate. The validation set has
only 200 records, so calibration conclusions require confirmation on later data.

## Explainability

Permutation importance measures the change in validation average precision when
one raw feature is shuffled. This is model-agnostic and keeps feature names at
their documented source level.

Local explanations are also model-agnostic. For each selected high-risk record,
one feature at a time is replaced with its training-set median or mode. The
resulting change in probability describes model sensitivity to that feature.
These effects are not causal and do not sum to the full prediction difference.

Correlated variables can divide or conceal permutation importance. Sensitive
attributes and proxy variables require fairness, legal, and governance review.
The prototype target is contract compliance rather than default within 90 days.

## Run

Generate splits and baseline results first, then optimize:

```powershell
.\.venv\Scripts\python.exe -m src.data_preparation
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.optimize
```

Optimized pipelines are saved as ignored local `.joblib` files. Structured model
results, feature importance, example explanations, Markdown findings, and figures
are written under `reports/`.
