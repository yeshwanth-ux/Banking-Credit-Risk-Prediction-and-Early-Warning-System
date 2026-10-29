"""Generate analyst-facing portfolio risk reports with Pandas."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.risk_reporting import PostgresRiskRepository, RiskRecordFilters, RiskRepository

RISK_ORDER = ("high", "medium", "low")
REPORT_COLUMNS = (
    "prediction_id",
    "scoring_run_reference",
    "reference_identifier",
    "loan_reference",
    "loan_status",
    "prediction_timestamp",
    "model_name",
    "model_version",
    "model_uri",
    "configuration_version",
    "default_probability",
    "risk_band",
    "warning_reasons",
)


@dataclass(frozen=True)
class GeneratedReports:
    output_directory: Path
    summary_csv: Path
    high_risk_csv: Path
    trends_csv: Path
    html_report: Path
    latest_account_count: int
    high_risk_account_count: int
    observed_period_count: int
    trend_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_directory": str(self.output_directory),
            "summary_csv": str(self.summary_csv),
            "high_risk_csv": str(self.high_risk_csv),
            "trends_csv": str(self.trends_csv),
            "html_report": str(self.html_report),
            "latest_account_count": self.latest_account_count,
            "high_risk_account_count": self.high_risk_account_count,
            "observed_period_count": self.observed_period_count,
            "trend_status": self.trend_status,
        }


def _records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    for column in REPORT_COLUMNS:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    frame = frame[list(REPORT_COLUMNS)].copy()
    frame["prediction_timestamp"] = pd.to_datetime(
        frame["prediction_timestamp"],
        utc=True,
        errors="coerce",
    )
    frame["default_probability"] = pd.to_numeric(
        frame["default_probability"],
        errors="coerce",
    )
    if frame["prediction_timestamp"].isna().any():
        raise ValueError("Report records contain invalid prediction timestamps")
    if frame["default_probability"].isna().any() or not frame[
        "default_probability"
    ].between(0, 1).all():
        raise ValueError("Report records contain invalid default probabilities")
    invalid_bands = sorted(set(frame["risk_band"].dropna()) - set(RISK_ORDER))
    if invalid_bands:
        raise ValueError("Report records contain invalid risk bands: " + ", ".join(invalid_bands))
    return frame


def latest_account_records(history: pd.DataFrame) -> pd.DataFrame:
    """Return one newest observed prediction for every external reference."""

    if history.empty:
        return history.copy()
    return (
        history.sort_values(["prediction_timestamp", "prediction_id"])
        .drop_duplicates("reference_identifier", keep="last")
        .sort_values(["risk_band", "default_probability"], ascending=[True, False])
        .reset_index(drop=True)
    )


def portfolio_summary(latest: pd.DataFrame) -> pd.DataFrame:
    """Calculate complete counts, proportions, and average probabilities by band."""

    total = len(latest)
    rows: list[dict[str, Any]] = []
    for band in RISK_ORDER:
        selected = latest.loc[latest["risk_band"].eq(band)]
        rows.append(
            {
                "risk_band": band,
                "account_count": len(selected),
                "portfolio_proportion": len(selected) / total if total else 0.0,
                "average_default_probability": (
                    float(selected["default_probability"].mean())
                    if not selected.empty
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _warning_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "No rule trigger; model threshold only"
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value or "No rule trigger; model threshold only"
    if not isinstance(value, list) or not value:
        return "No rule trigger; model threshold only"
    reasons: list[str] = []
    for warning in value:
        if isinstance(warning, dict):
            reason = warning.get("reason") or warning.get("code")
            if reason:
                reasons.append(str(reason))
        else:
            reasons.append(str(warning))
    return " | ".join(reasons) or "No rule trigger; model threshold only"


def high_risk_accounts(latest: pd.DataFrame) -> pd.DataFrame:
    """Build the restricted investigation queue for latest High Risk accounts."""

    columns = [
        "reference_identifier",
        "loan_reference",
        "loan_status",
        "prediction_timestamp",
        "default_probability",
        "warning_reasons",
        "model_name",
        "model_version",
        "configuration_version",
    ]
    selected = latest.loc[latest["risk_band"].eq("high"), columns].copy()
    selected["warning_reasons"] = selected["warning_reasons"].map(_warning_text)
    return selected.sort_values(
        ["default_probability", "prediction_timestamp"],
        ascending=[False, False],
    ).reset_index(drop=True)


def scoring_trends(history: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Summarize observed daily scoring states without inventing historical periods."""

    columns = [
        "scoring_period",
        "total_accounts",
        "high_count",
        "medium_count",
        "low_count",
        "high_proportion",
        "medium_proportion",
        "low_proportion",
        "average_default_probability",
        "data_status",
    ]
    if history.empty:
        return pd.DataFrame(columns=columns), "no_observed_scoring_periods"
    daily = history.copy()
    daily["scoring_period"] = daily["prediction_timestamp"].dt.floor("D")
    daily = (
        daily.sort_values(["prediction_timestamp", "prediction_id"])
        .drop_duplicates(["scoring_period", "reference_identifier"], keep="last")
    )
    period_count = daily["scoring_period"].nunique()
    status = (
        "observed_scoring_history"
        if period_count >= 2
        else "insufficient_observed_periods"
    )
    rows: list[dict[str, Any]] = []
    for period, group in daily.groupby("scoring_period", sort=True):
        total = len(group)
        counts = group["risk_band"].value_counts()
        rows.append(
            {
                "scoring_period": period,
                "total_accounts": total,
                "high_count": int(counts.get("high", 0)),
                "medium_count": int(counts.get("medium", 0)),
                "low_count": int(counts.get("low", 0)),
                "high_proportion": counts.get("high", 0) / total,
                "medium_proportion": counts.get("medium", 0) / total,
                "low_proportion": counts.get("low", 0) / total,
                "average_default_probability": float(
                    group["default_probability"].mean()
                ),
                "data_status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns), status


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _html_report(
    summary: pd.DataFrame,
    high_risk: pd.DataFrame,
    trends: pd.DataFrame,
    *,
    generated_at: datetime,
    trend_status: str,
) -> str:
    if trend_status == "observed_scoring_history":
        trend_statement = (
            "The trend table contains two or more observed scoring periods. Changes are "
            "descriptive and must not be interpreted as causal or as validated portfolio "
            "deterioration without checking population, model, and data consistency."
        )
    elif trend_status == "insufficient_observed_periods":
        trend_statement = (
            "Only one observed scoring period is available. No historical trend conclusion "
            "is supported. No simulated period has been presented as history."
        )
    else:
        trend_statement = (
            "No observed scoring periods are available. No trend conclusion is supported "
            "and no simulated trend was generated."
        )
    high_risk_table = (
        high_risk.to_html(index=False, border=0)
        if not high_risk.empty
        else '<p class="empty">No latest High Risk accounts in the selected scope.</p>'
    )
    trends_table = (
        trends.to_html(
            index=False,
            border=0,
            float_format=lambda value: f"{value:.4f}",
        )
        if not trends.empty
        else '<p class="empty">No observed scoring history is available.</p>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Portfolio Credit Risk Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 1100px; color: #17202a; }}
    h1, h2 {{ color: #17365d; }}
    .notice {{ background: #fff4ce; border-left: 5px solid #d69e00; padding: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #ccd4dd; padding: .55rem; text-align: left; }}
    th {{ background: #eaf0f6; }}
    .empty {{ color: #5f6b76; font-style: italic; }}
  </style>
</head>
<body>
  <h1>Portfolio Credit Risk and Early-Warning Report</h1>
  <p><strong>Generated:</strong> {html.escape(generated_at.isoformat())}</p>
  <div class="notice"><strong>Research prototype:</strong> probabilities describe the configured
  prototype target. Risk bands and warnings support investigation and do not constitute an
  automated lending decision.</div>
  <h2>Latest portfolio risk summary</h2>
  {summary.to_html(index=False, border=0, float_format=lambda value: f"{value:.4f}")}
  <h2>High Risk accounts requiring investigation</h2>
  {high_risk_table}
  <h2>Observed scoring-period trends</h2>
  <p>{html.escape(trend_statement)}</p>
  {trends_table}
  <h2>Analyst verification checklist</h2>
  <ol>
    <li>Confirm the report date range, account population, loan status, model version,
    and configuration version.</li>
    <li>Reconcile High Risk accounts with authorized loan and repayment systems before
    taking action.</li>
    <li>Review probability drivers separately from rule-based warning reasons and confirm
    every input was available at prediction time.</li>
    <li>Check missing data, source freshness, model calibration, fairness monitoring, and
    threshold governance.</li>
    <li>Treat period changes as descriptive observations; investigate population or model
    changes before claiming a historical trend.</li>
  </ol>
</body>
</html>
"""


def generate_portfolio_reports(
    output_directory: Path,
    *,
    filters: RiskRecordFilters | None = None,
    repository: RiskRepository | None = None,
    records: list[dict[str, Any]] | None = None,
    generated_at: datetime | None = None,
) -> GeneratedReports:
    """Export summary, High Risk queue, observed trends, and readable HTML."""

    selected_filters = filters or RiskRecordFilters()
    selected_filters.validate()
    if records is None:
        selected_repository = repository or PostgresRiskRepository()
        records = selected_repository.fetch_scoring_history(selected_filters)
    history = _records_frame(records)
    latest = latest_account_records(history)
    summary = portfolio_summary(latest)
    high_risk = high_risk_accounts(latest)
    trends, trend_status = scoring_trends(history)
    timestamp = generated_at or datetime.now(tz=UTC)

    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "portfolio_risk_summary.csv"
    high_risk_path = output_directory / "high_risk_accounts.csv"
    trends_path = output_directory / "risk_trends.csv"
    html_path = output_directory / "portfolio_risk_report.html"
    _atomic_csv(summary, summary_path)
    _atomic_csv(high_risk, high_risk_path)
    _atomic_csv(trends, trends_path)
    temporary_html = html_path.with_suffix(".html.tmp")
    temporary_html.write_text(
        _html_report(
            summary,
            high_risk,
            trends,
            generated_at=timestamp,
            trend_status=trend_status,
        ),
        encoding="utf-8",
    )
    temporary_html.replace(html_path)
    return GeneratedReports(
        output_directory=output_directory.resolve(),
        summary_csv=summary_path.resolve(),
        high_risk_csv=high_risk_path.resolve(),
        trends_csv=trends_path.resolve(),
        html_report=html_path.resolve(),
        latest_account_count=len(latest),
        high_risk_account_count=len(high_risk),
        observed_period_count=history["prediction_timestamp"].dt.floor("D").nunique(),
        trend_status=trend_status,
    )
