"""Build the reproducible Day 2 exploratory analysis notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "credit_risk_analysis.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook(execute: bool = False) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        markdown(
            """
# South German Credit exploratory data analysis

This notebook profiles the corrected South German Credit dataset from the UCI
Machine Learning Repository (dataset 573; DOI: 10.24432/C5QG88). The dataset is
licensed under CC BY 4.0.

The analysis describes data quality and class structure. It does not establish
production suitability or a validated probability of default.
"""
        ),
        markdown(
            """
## Reproducibility and source

The loader downloads the official UCI archive, verifies fixed SHA-256 checksums,
preserves the source files in the ignored `data/raw` directory, and creates a
normalized CSV with descriptive English field names.
"""
        ),
        code(
            """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.download_data import download_dataset, validate_dataset

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)
"""
        ),
        code(
            """
dataset_path = download_dataset(ROOT / "data" / "raw")
data = pd.read_csv(dataset_path)
validate_dataset(data)

print(f"Rows: {len(data):,}")
print(f"Columns: {data.shape[1]}")
data.head()
"""
        ),
        markdown("## Schema, missing values, and duplicate records"),
        code(
            """
schema = pd.DataFrame({
    "dtype": data.dtypes.astype(str),
    "missing_count": data.isna().sum(),
    "missing_percent": data.isna().mean().mul(100).round(2),
    "unique_values": data.nunique(),
})
schema
"""
        ),
        code(
            """
quality_summary = pd.Series({
    "rows": len(data),
    "columns": data.shape[1],
    "duplicate_rows": int(data.duplicated().sum()),
    "missing_cells": int(data.isna().sum().sum()),
    "constant_columns": int((data.nunique() <= 1).sum()),
}, name="value")
quality_summary.to_frame()
"""
        ),
        code(
            """
missing = data.isna().mean().mul(100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 4))
missing.plot(kind="bar", ax=ax, color="#315b7d")
ax.set(title="Missing values by field", xlabel="Field", ylabel="Missing values (%)")
ax.set_ylim(0, max(1, missing.max() * 1.15))
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.show()
"""
        ),
        markdown("## Target definition and class balance"),
        markdown(
            """
The UCI outcome `credit_risk` uses `0 = bad` and `1 = good`. The modeling label
is `bad_credit = 1 - credit_risk`, so the positive class means failed contract
compliance. Both label columns must be excluded from model features.
"""
        ),
        code(
            """
target_counts = data["bad_credit"].value_counts().sort_index().rename(index={0: "good", 1: "bad"})
target_summary = pd.DataFrame({
    "count": target_counts,
    "percent": (target_counts / len(data) * 100).round(1),
})
target_summary
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(6, 4))
target_counts.plot(kind="bar", ax=ax, color=["#3a7d44", "#b23a48"])
ax.set(title="Credit outcome class balance", xlabel="Outcome", ylabel="Records")
ax.tick_params(axis="x", rotation=0)
for patch in ax.patches:
    label_position = (patch.get_x() + patch.get_width() / 2, patch.get_height())
    ax.annotate(
        f"{int(patch.get_height())}", label_position, ha="center", va="bottom"
    )
plt.tight_layout()
plt.show()
"""
        ),
        markdown("## Numeric distributions"),
        code(
            """
numeric_fields = ["duration_months", "credit_amount_dm", "age_years"]
data[numeric_fields].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for field, ax in zip(numeric_fields, axes):
    sns.histplot(data=data, x=field, hue="bad_credit", bins=25, element="step", stat="density",
                 common_norm=False, ax=ax, palette={0: "#3a7d44", 1: "#b23a48"})
    ax.set_title(field.replace("_", " ").title())
plt.tight_layout()
plt.show()
"""
        ),
        markdown("## Categorical and ordinal distributions"),
        code(
            """
categorical_fields = [
    "checking_account_status", "credit_history", "purpose", "savings",
    "employment_duration", "installment_rate", "housing", "job",
]

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
for field, ax in zip(categorical_fields, axes.flat):
    order = sorted(data[field].unique())
    sns.countplot(data=data, x=field, order=order, ax=ax, color="#4c78a8")
    ax.set_title(field.replace("_", " ").title())
plt.tight_layout()
plt.show()
"""
        ),
        markdown("## Outcome rates across selected indicators"),
        code(
            """
rate_fields = ["checking_account_status", "credit_history", "savings", "duration_months"]
rate_tables = {}
for field in rate_fields:
    rate_tables[field] = (
        data.groupby(field, observed=True)["bad_credit"]
        .agg(records="size", bad_credit_rate="mean")
        .assign(bad_credit_rate=lambda frame: frame["bad_credit_rate"].round(3))
    )

rate_tables["checking_account_status"]
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for field, ax in zip(["checking_account_status", "credit_history", "savings"], axes):
    rates = rate_tables[field].reset_index()
    sns.barplot(data=rates, x=field, y="bad_credit_rate", ax=ax, color="#b23a48")
    ax.set_ylim(0, 1)
    ax.set_title(f"Bad-credit rate by {field.replace('_', ' ')}")
plt.tight_layout()
plt.show()
"""
        ),
        markdown("## Range and consistency checks"),
        code(
            """
checks = {
    "duration_positive": bool((data["duration_months"] > 0).all()),
    "amount_positive": bool((data["credit_amount_dm"] > 0).all()),
    "age_positive": bool((data["age_years"] > 0).all()),
    "credit_risk_binary": set(data["credit_risk"]) == {0, 1},
    "bad_credit_binary": set(data["bad_credit"]) == {0, 1},
    "target_inverse_consistent": bool((data["bad_credit"] == 1 - data["credit_risk"]).all()),
}
pd.Series(checks, name="passed").to_frame()
"""
        ),
        markdown(
            """
## Data quality, leakage, and fairness concerns

- The source declares no missing values, but missingness cannot reveal whether
  categories such as unknown/no account conceal unobserved information.
- Duplicate rows are not automatically invalid because the dataset has no unique
  customer or loan identifier. Exact duplicates should be reviewed, not silently
  removed.
- The 30% bad-credit share reflects deliberate oversampling and is not a current
  population default rate.
- `credit_amount_dm` was transformed by an undocumented monotonic function.
- `credit_history` includes previous or concurrent contracts, but timestamps are
  unavailable. Its real-time availability cannot be proven.
- `personal_status_sex`, `age_years`, and `foreign_worker` are fairness-relevant.
  They require governance and should not drive automated lending decisions.
- The data contain no repayment timeline, so they cannot support early-warning
  trends, time-based validation, or the planned 90-day default target.
"""
        ),
        markdown(
            """
## Day 2 conclusion

The dataset is suitable for a reproducible educational classification baseline
and for testing the project pipeline. It is unsuitable for current portfolio
risk estimation or production lending decisions. A later project phase needs a
dated, appropriately licensed repayment dataset to implement 90-day default and
early-warning behavior.
"""
        ),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if execute:
        client = NotebookClient(notebook, timeout=180, kernel_name="python3", allow_errors=False)
        client.execute(cwd=str(ROOT))

    nbf.write(notebook, OUTPUT)
    action = "Executed and wrote" if execute else "Wrote"
    print(f"{action} {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    build_notebook(execute=parser.parse_args().execute)
