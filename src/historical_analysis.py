"""Analyze historical risk patterns and generate reproducible figures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.database import DatabaseSettings
from src.feature_engineering import engineer_prototype_features, load_prototype_feature_source

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "historical_patterns.md"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


@dataclass(frozen=True)
class AnalysisResult:
    rows: int
    default_rate: float
    extreme_value_counts: dict[str, int]
    redundant_pairs: list[tuple[str, str, float]]
    generated_figures: list[str]
    unavailable_analyses: list[str]


def default_rate_by_quantile(
    data: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 5,
) -> pd.DataFrame:
    """Return observed target rate in duplicate-safe quantile groups."""

    values = pd.to_numeric(data[feature], errors="coerce")
    valid = values.notna() & data[target].notna()
    if valid.sum() < bins:
        raise ValueError(f"Not enough valid values to bin {feature}")
    groups = pd.qcut(values.loc[valid], q=bins, duplicates="drop")
    result = data.loc[valid, [target]].groupby(groups, observed=True)[target].agg(
        default_rate="mean", records="size"
    )
    result.index = result.index.astype(str)
    return result.reset_index(names="band")


def extreme_value_counts(data: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Count values outside 1.5 interquartile ranges for each numeric feature."""

    counts: dict[str, int] = {}
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        first, third = values.quantile([0.25, 0.75])
        spread = third - first
        lower, upper = first - 1.5 * spread, third + 1.5 * spread
        counts[column] = int(((values < lower) | (values > upper)).sum())
    return counts


def find_redundant_pairs(
    data: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.90,
) -> list[tuple[str, str, float]]:
    """Return unique feature pairs at or above an absolute correlation threshold."""

    correlation = data[columns].apply(pd.to_numeric, errors="coerce").corr()
    pairs: list[tuple[str, str, float]] = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((left, right, float(value)))
    return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)


def _save_rate_chart(table: pd.DataFrame, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(table["band"], table["default_rate"], color="#355C7D")
    axis.set_title(title)
    axis.set_ylabel("Observed bad-credit rate")
    axis.set_xlabel("")
    axis.set_ylim(0, min(1.0, max(0.4, float(table["default_rate"].max()) * 1.2)))
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_distribution_chart(data: pd.DataFrame, columns: list[str], path: Path) -> None:
    figure, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
    axes_array = np.atleast_1d(axes)
    for axis, column in zip(axes_array, columns, strict=True):
        axis.hist(pd.to_numeric(data[column], errors="coerce").dropna(), bins=25, color="#6C5B7B")
        axis.set_title(column.replace("_", " ").title())
        axis.set_ylabel("Records")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def generate_historical_analysis(
    report_path: Path = DEFAULT_REPORT,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    settings: DatabaseSettings | None = None,
) -> AnalysisResult:
    data = engineer_prototype_features(load_prototype_feature_source(settings))
    if data.empty:
        raise ValueError("Trusted prototype dataset is empty")
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    amount_rates = default_rate_by_quantile(data, "credit_amount_dm", "bad_credit")
    amount_path = figure_dir / "default_rate_by_credit_amount.png"
    _save_rate_chart(amount_rates, "Bad-credit rate by credit amount", amount_path)
    generated.append(amount_path.name)

    behavior = (
        data.groupby("installment_rate", observed=True)["bad_credit"]
        .agg(default_rate="mean", records="size")
        .reset_index()
        .rename(columns={"installment_rate": "band"})
    )
    behavior["band"] = behavior["band"].map(lambda value: f"Code {value}")
    behavior_path = figure_dir / "default_rate_by_installment_burden.png"
    _save_rate_chart(behavior, "Bad-credit rate by installment-burden code", behavior_path)
    generated.append(behavior_path.name)

    numeric = ["duration_months", "credit_amount_dm", "age_years", "credit_amount_per_month"]
    distribution_path = figure_dir / "feature_distributions.png"
    _save_distribution_chart(data, numeric, distribution_path)
    generated.append(distribution_path.name)

    extremes = extreme_value_counts(data, numeric)
    redundant = find_redundant_pairs(data, numeric)
    numeric_relationships = (
        data[numeric + ["bad_credit"]]
        .apply(pd.to_numeric, errors="coerce")
        .corr()["bad_credit"]
        .drop("bad_credit")
        .sort_values(key=abs, ascending=False)
    )
    unavailable = [
        "Income-level default rates: the research source has no income field.",
        "Transaction-derived repayment behavior: the source has no dated repayments.",
        "Time-based splits: the source has no reliable event or prediction dates.",
    ]

    lines = [
        "# Historical risk patterns",
        "",
        f"**Records analyzed:** {len(data):,}",
        f"**Observed bad-credit rate:** {data['bad_credit'].mean():.2%}",
        "",
        "## Feature relationships",
        "",
        "| Numeric feature | Correlation with bad credit |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {value:.4f} |" for name, value in numeric_relationships.items())
    lines.extend(["", "## Extreme values", "", "| Feature | IQR outliers |", "|---|---:|"])
    lines.extend(f"| `{name}` | {count:,} |" for name, count in extremes.items())
    lines.extend(["", "## Highly correlated or redundant variables", ""])
    if redundant:
        lines.extend(f"- `{left}` and `{right}`: {value:.4f}" for left, right, value in redundant)
    else:
        lines.append("No numeric feature pairs met the absolute 0.90 correlation threshold.")
    lines.extend(["", "## Figures", ""])
    lines.extend(f"- `reports/figures/{name}`" for name in generated)
    lines.extend(["", "## Data limitations", ""])
    lines.extend(f"- {item}" for item in unavailable)
    lines.extend(
        [
            "",
            "The installment-burden chart uses the source's coded installment-rate band. "
            "It is not a measured repayment history. Associations are descriptive and do "
            "not establish causation.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return AnalysisResult(
        rows=len(data),
        default_rate=float(data["bad_credit"].mean()),
        extreme_value_counts=extremes,
        redundant_pairs=redundant,
        generated_figures=generated,
        unavailable_analyses=unavailable,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()
    result = generate_historical_analysis(args.report, args.figure_dir)
    print(f"Analyzed {result.rows:,} records at {result.default_rate:.2%} bad credit")
    print(f"Generated figures: {len(result.generated_figures)}")


if __name__ == "__main__":
    main()
