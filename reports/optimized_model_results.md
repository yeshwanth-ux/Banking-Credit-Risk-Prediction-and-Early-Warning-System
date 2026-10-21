# Optimized model results and explainability

**Cross-validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
**Search objective:** training-fold average precision (PR-AUC)
**Final comparison split:** validation
**Held-out test used:** no
**Selected validation champion:** `logistic_regression`

## Optimized model comparison

| Model | CV PR-AUC | Validation ROC-AUC | Validation PR-AUC | Precision | Recall | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6029 | 0.7661 | 0.6166 | 0.4574 | 0.7167 | 0.5584 | 0.2061 | 0.1811 |
| Random Forest | 0.6183 | 0.7764 | 0.6106 | 0.5417 | 0.6500 | 0.5909 | 0.1859 | 0.1318 |

## Change from baseline

| Model | ROC-AUC change | PR-AUC change |
|---|---:|---:|
| Logistic Regression | +0.0100 | +0.0359 |
| Random Forest | -0.0132 | -0.0211 |

## Primary predictive features

Permutation importance measures the validation PR-AUC decrease after one raw feature is shuffled. Larger positive values indicate greater model reliance on that feature.

| Feature | Mean importance | Standard deviation |
|---|---:|---:|
| `checking_account_status` | 0.1390 | 0.0298 |
| `duration_months` | 0.0688 | 0.0299 |
| `savings` | 0.0503 | 0.0194 |
| `purpose` | 0.0460 | 0.0155 |
| `credit_history` | 0.0455 | 0.0214 |
| `other_installment_plans` | 0.0431 | 0.0155 |
| `credit_amount_dm` | 0.0275 | 0.0229 |
| `personal_status_sex` | 0.0269 | 0.0213 |
| `telephone` | 0.0233 | 0.0135 |
| `installment_rate` | 0.0199 | 0.0206 |

## Explanation method and limitations

Local explanations replace one raw feature at a time with its training-set median or mode and measure the change in predicted probability. They show model sensitivity for the selected record, not causality, approval reasons, or additive probability decompositions.

Permutation results can be unstable when features are correlated because related variables can substitute for one another. Calibration estimates use only 200 validation records and require confirmation on later data. The research label is contract compliance, not default within 90 days. Sensitive or proxy attributes require fairness and governance review before deployment.

The champion is selected on validation PR-AUC. The test split remains sealed until model settings and the decision threshold are frozen.
