# Data ingestion and validation

The Day 4 pipeline reads `data/raw/banking_data.csv`, normalizes its headers,
validates every source field against the Day 2 data dictionary, and upserts valid
observations into PostgreSQL. It does not invent application dates, repayment
dates, interest rates, identifiers, or currency values that the South German
Credit dataset does not provide.

## Rules

- All 22 documented fields are required; the source declares no missing values.
- Numeric fields must be finite integers because all official source fields are
  integer-coded.
- Duration and credit amount must be positive, and age must be 18 through 120.
- Categorical values must match the official UCI code ranges.
- `credit_risk` and `bad_credit` must be binary and exact inverses.
- A SHA-256 identifier over canonical field values detects duplicate source rows
  and provides an idempotent PostgreSQL upsert key.
- Optional reusable financial validation covers nonnegative amounts, positive
  principal, interest rates from 0% through 100%, ISO `YYYY-MM-DD` dates, and
  application/origination/maturity/repayment ordering.

Missing or malformed records are written to `ingestion_rejections` with their
source row number and machine-readable error codes. `ingestion_runs` records the
source checksum, status, accepted/rejected totals, insert/update totals, and any
processing error. Database writes for observations, rejections, and successful
run completion use one transaction.

## Run

Apply migrations and ingest the default source file:

```powershell
.\.venv\Scripts\python.exe -m src.migrations up
.\.venv\Scripts\python.exe -m src.ingestion
```

Specify another CSV or increase logging when needed:

```powershell
.\.venv\Scripts\python.exe -m src.ingestion data/raw/banking_data.csv --log-level DEBUG
```

The command prints a JSON summary. Re-running the same valid source updates the
existing observations rather than creating duplicates.
