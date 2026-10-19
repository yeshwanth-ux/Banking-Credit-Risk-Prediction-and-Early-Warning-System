# Baseline credit risk model results

Both models use class weights and a fixed 0.50 threshold. They were fitted on the training split and evaluated on the validation split. The test split was not read or evaluated.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | False alerts | Missed defaults |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7561 | 0.5807 | 0.4944 | 0.7333 | 0.5906 | 45 | 16 |
| Random Forest | 0.7896 | 0.6317 | 0.5692 | 0.6167 | 0.5920 | 28 | 23 |

## Operational trade-off

`logistic_regression` misses the fewest bad-credit cases at this threshold; higher recall reduces missed defaults but can send more good-credit cases to manual review.

`random_forest` produces the fewest false alerts at this threshold; fewer false positives reduce review workload but may increase missed bad-credit cases.

Model selection and any threshold change belong to validation. Final test evaluation remains deferred until the model and threshold are frozen.
