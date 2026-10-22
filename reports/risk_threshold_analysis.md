# Early-warning threshold analysis

**Configuration:** `risk-classification-1.0.0`
**Model:** `optimized-logistic-regression-1.0.0`
**Selection data:** validation only
**Held-out test used:** no

## Frozen risk bands

- Low: `probability < 0.44`
- Medium: `0.44 <= probability < 0.84`
- High: `probability >= 0.84`

## Business cost assumptions

The medium threshold represents a manual-review decision. A missed bad-credit case is assigned five cost units and an unnecessary review one cost unit.

The high threshold represents a more intensive intervention. An unnecessary intervention is assigned four cost units and a missed high-band case one cost unit. This produces a narrow high band; severe rule triggers can independently escalate the final warning to High.

| Decision | Threshold | FP | FN | Precision | Recall | Weighted cost |
|---|---:|---:|---:|---:|---:|---:|
| Medium review | 0.44 | 58 | 9 | 0.4679 | 0.8500 | 103.0 |
| High intervention | 0.84 | 0 | 54 | 1.0000 | 0.1000 | 54.0 |

## Limitations

The frozen model predicts the prototype `bad_credit` outcome, not a dated 90-day probability of default. Costs are initial assumptions for portfolio demonstration and require business, risk, fairness, and compliance review. Validation has 200 records, so thresholds require monitoring and later-data confirmation before operational use.
