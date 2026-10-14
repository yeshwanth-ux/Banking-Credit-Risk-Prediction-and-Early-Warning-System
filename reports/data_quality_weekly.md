# Weekly dataset quality summary

**Report date:** 2026-10-14
**Overall status:** PASS
**Checks passed:** 8/8

## Dataset coverage

| Dataset | Rows | Loans | Bad-credit rate |
|---|---:|---:|---:|
| Operational point-in-time dataset | 0 | 0 | n/a |
| South German Credit prototype | 1,000 | n/a | 30.00% |

## Quality checks

| Check | Status | Failed rows | Rule |
|---|---|---:|---|
| `operational_unique_key` | PASS | 0 | Each operational record has a unique loan and prediction date. |
| `operational_required_fields` | PASS | 0 | Operational identifiers and required loan values are populated. |
| `operational_valid_ranges` | PASS | 0 | Operational numeric features remain within valid ranges. |
| `operational_valid_keys` | PASS | 0 | Every prediction date resolves to a valid loan and customer. |
| `operational_prediction_timing` | PASS | 0 | No prediction precedes its loan application. |
| `prototype_unique_key` | PASS | 0 | Each prototype record has a unique deterministic source key. |
| `prototype_required_fields` | PASS | 0 | Required prototype model fields are populated. |
| `prototype_valid_ranges` | PASS | 0 | Prototype numeric values and label remain within valid ranges. |

## Interpretation

The operational dataset contains one row per loan and prediction date. Repayment aggregates include only installments due on or before that date, and payments made after it are excluded from features.

The South German Credit prototype has no event dates or repayment history. Its `bad_credit` outcome supports exploratory classification only and must not be interpreted as the project's 90-day default target.
