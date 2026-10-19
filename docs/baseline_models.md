# Baseline credit risk models

Day 8 trains two Scikit-learn baselines on the South German Credit research
prototype:

- Logistic Regression with `class_weight="balanced"`; and
- Random Forest with `class_weight="balanced_subsample"`.

Both models contain their own preprocessing pipeline. Numerical median
imputation, scaling, categorical imputation, and one-hot encoding are fitted
only from the 600 training records. The models use a fixed 0.50 threshold and
are compared on the 200 validation records. The test file is not opened by the
training command and remains reserved for final evaluation after model and
threshold selection.

## Metrics

- ROC-AUC measures ranking across all classification thresholds.
- PR-AUC emphasizes performance on the positive bad-credit class.
- Precision measures how many flagged records are actually bad credit.
- Recall measures how many bad-credit records are detected.
- F1 is the harmonic mean of precision and recall.
- Confusion matrices expose false positives and false negatives at threshold 0.50.

A false positive sends a good-credit record to unnecessary review. A false
negative misses a bad-credit record. Raising recall usually reduces missed
defaults while increasing review workload; raising precision typically reduces
false alerts while risking more missed defaults. Threshold selection is an
operational decision and is deferred until the validation comparison is reviewed.

## Run

Generate Day 7 split files first, then train the baselines:

```powershell
.\.venv\Scripts\python.exe -m src.data_preparation
.\.venv\Scripts\python.exe -m src.train
```

The command writes complete fitted pipelines to ignored local `.joblib` files in
`models/`. Structured validation results are saved as JSON and CSV, with a
Markdown comparison and confusion-matrix figure under `reports/`.

These models predict the prototype `bad_credit` contract-compliance outcome.
They are not calibrated estimates of default within 90 days and are not approved
for lending decisions.
