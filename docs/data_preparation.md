# Historical analysis and model data preparation

Day 7 creates reproducible historical analysis and leakage-safe model inputs.

## Historical patterns

Run the analysis with:

```powershell
.\.venv\Scripts\python.exe -m src.historical_analysis
```

It generates `reports/historical_patterns.md` and three figures covering credit
amount default rates, the source installment-burden code, and numeric feature
distributions. The report includes target correlations, IQR extreme-value counts,
and numeric pairs whose absolute Pearson correlation is at least 0.90.

The South German Credit source has no income or dated repayment transactions.
The report records that income-level and transaction-derived repayment analyses
are unavailable. The installment chart uses a coded affordability band and is
not presented as observed payment history.

## Dataset splits

Run preparation with:

```powershell
.\.venv\Scripts\python.exe -m src.data_preparation
```

The splitter uses chronological 60/20/20 partitions when a fully populated date
column has at least three distinct dates. Otherwise it uses a reproducible,
stratified 60/20/20 split. The current research dataset has no reliable dates,
so it uses the stratified path with random seed 42.

The row-level assignments and split datasets are local ignored files under
`data/processed/`. The non-sensitive definition, counts, target rates, method,
seed, and feature lists are saved to `reports/dataset_split_summary.json`.

## Preprocessing

Numerical fields use median imputation followed by standard scaling. Categorical
fields use most-frequent imputation and one-hot encoding with unknown categories
ignored. The `ColumnTransformer` is fit only on training rows and saved locally
as the ignored artifact `models/preprocessor.joblib`. Validation and test rows
never influence imputation values, scaling statistics, or encoded categories.

Identifiers and `bad_credit` are excluded from preprocessing. `bad_credit` is
the prototype contract-compliance outcome and is not a 90-day default label.
