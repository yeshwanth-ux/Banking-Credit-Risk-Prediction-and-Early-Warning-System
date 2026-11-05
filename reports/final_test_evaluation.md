# Final untouched-test evaluation

**Evaluation status:** completed once<br>
**Evaluated at:** 2026-11-05T11:08:37.681567+00:00<br>
**Dataset:** South German Credit research prototype<br>
**Target:** `bad_credit`<br>
**Test rows:** 200<br>
**Decision threshold:** 0.44 (selected on validation data)<br>
**Model version:** optimized-logistic-regression-1.0.0

## Test metrics

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8143 |
| PR-AUC | 0.6594 |
| Precision | 0.4842 |
| Recall | 0.7667 |
| F1 | 0.5935 |
| Brier score | 0.1739 |
| Expected calibration error | 0.1418 |

## Confusion matrix

| | Predicted good | Predicted bad |
|---|---:|---:|
| Actual good | 91 | 49 |
| Actual bad | 14 | 46 |

## Interpretation

This is the only final evaluation on the held-out test split. The threshold was
frozen from validation analysis before test access and was not tuned on these
results. The dataset models historical contract non-compliance, not a current
90-day probability of default. The small, oversampled 1973-1975 dataset cannot
establish production performance, calibration, fairness, or regulatory fitness.
