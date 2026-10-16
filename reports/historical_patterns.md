# Historical risk patterns

**Records analyzed:** 1,000
**Observed bad-credit rate:** 30.00%

## Feature relationships

| Numeric feature | Correlation with bad credit |
|---|---:|
| `duration_months` | 0.2149 |
| `credit_amount_dm` | 0.1547 |
| `age_years` | -0.0913 |
| `credit_amount_per_month` | 0.0186 |

## Extreme values

| Feature | IQR outliers |
|---|---:|
| `duration_months` | 70 |
| `credit_amount_dm` | 72 |
| `age_years` | 23 |
| `credit_amount_per_month` | 45 |

## Highly correlated or redundant variables

No numeric feature pairs met the absolute 0.90 correlation threshold.

## Figures

- `reports/figures/default_rate_by_credit_amount.png`
- `reports/figures/default_rate_by_installment_burden.png`
- `reports/figures/feature_distributions.png`

## Data limitations

- Income-level default rates: the research source has no income field.
- Transaction-derived repayment behavior: the source has no dated repayments.
- Time-based splits: the source has no reliable event or prediction dates.

The installment-burden chart uses the source's coded installment-rate band. It is not a measured repayment history. Associations are descriptive and do not establish causation.
