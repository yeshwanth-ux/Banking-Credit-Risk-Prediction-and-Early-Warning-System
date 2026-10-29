"""Tests for analyst-facing Pandas risk reports."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.reporting import generate_portfolio_reports


def report_records() -> list[dict[str, Any]]:
    base = {
        "scoring_run_reference": "run-1",
        "loan_reference": None,
        "loan_status": "active",
        "model_name": "CreditRiskEarlyWarning",
        "model_version": "1",
        "model_uri": "models:/CreditRiskEarlyWarning@champion",
        "configuration_version": "risk-classification-1.0.0",
    }
    return [
        {
            **base,
            "prediction_id": 1,
            "reference_identifier": "ACCOUNT-A",
            "prediction_timestamp": datetime(2026, 10, 27, 2, tzinfo=UTC),
            "default_probability": 0.55,
            "risk_band": "medium",
            "warning_reasons": [{"reason": "One missed installment"}],
        },
        {
            **base,
            "prediction_id": 2,
            "reference_identifier": "ACCOUNT-B",
            "prediction_timestamp": datetime(2026, 10, 27, 2, tzinfo=UTC),
            "default_probability": 0.20,
            "risk_band": "low",
            "warning_reasons": [],
        },
        {
            **base,
            "scoring_run_reference": "run-2",
            "prediction_id": 3,
            "reference_identifier": "ACCOUNT-A",
            "prediction_timestamp": datetime(2026, 10, 28, 2, tzinfo=UTC),
            "default_probability": 0.91,
            "risk_band": "high",
            "warning_reasons": [{"reason": "Severe delinquency <verify>"}],
        },
        {
            **base,
            "scoring_run_reference": "run-2",
            "prediction_id": 4,
            "reference_identifier": "ACCOUNT-B",
            "prediction_timestamp": datetime(2026, 10, 28, 2, tzinfo=UTC),
            "default_probability": 0.60,
            "risk_band": "medium",
            "warning_reasons": [{"code": "increasing_debt_burden"}],
        },
        {
            **base,
            "scoring_run_reference": "run-2",
            "prediction_id": 5,
            "reference_identifier": "ACCOUNT-C",
            "prediction_timestamp": datetime(2026, 10, 28, 2, tzinfo=UTC),
            "default_probability": 0.88,
            "risk_band": "high",
            "warning_reasons": [],
        },
    ]


def test_exports_summary_high_risk_trends_and_readable_html(tmp_path: Path) -> None:
    generated = generate_portfolio_reports(
        tmp_path,
        records=report_records(),
        generated_at=datetime(2026, 10, 29, tzinfo=UTC),
    )

    summary = pd.read_csv(generated.summary_csv)
    high_risk = pd.read_csv(generated.high_risk_csv)
    trends = pd.read_csv(generated.trends_csv)
    document = generated.html_report.read_text(encoding="utf-8")

    counts = dict(zip(summary["risk_band"], summary["account_count"], strict=True))
    proportions = dict(
        zip(summary["risk_band"], summary["portfolio_proportion"], strict=True)
    )
    assert counts == {"high": 2, "medium": 1, "low": 0}
    assert sum(proportions.values()) == 1.0
    assert list(high_risk["reference_identifier"]) == ["ACCOUNT-A", "ACCOUNT-C"]
    assert high_risk.loc[0, "default_probability"] == 0.91
    assert "Severe delinquency <verify>" in high_risk.loc[0, "warning_reasons"]
    assert "model threshold only" in high_risk.loc[1, "warning_reasons"]
    assert list(trends["total_accounts"]) == [2, 3]
    assert list(trends["high_count"]) == [0, 2]
    assert set(trends["data_status"]) == {"observed_scoring_history"}
    assert generated.latest_account_count == 3
    assert generated.high_risk_account_count == 2
    assert generated.observed_period_count == 2
    assert generated.trend_status == "observed_scoring_history"
    assert "descriptive" in document
    assert "must not be interpreted as causal" in document
    assert "Analyst verification checklist" in document
    assert "Severe delinquency &lt;verify&gt;" in document


def test_single_observed_period_does_not_claim_historical_trend(tmp_path: Path) -> None:
    one_period = report_records()[:2]

    generated = generate_portfolio_reports(tmp_path, records=one_period)
    trends = pd.read_csv(generated.trends_csv)
    document = generated.html_report.read_text(encoding="utf-8")

    assert generated.observed_period_count == 1
    assert generated.trend_status == "insufficient_observed_periods"
    assert set(trends["data_status"]) == {"insufficient_observed_periods"}
    assert "No historical trend conclusion is supported" in document
    assert "No simulated period has been presented as history" in document


def test_empty_history_exports_complete_zero_summary(tmp_path: Path) -> None:
    generated = generate_portfolio_reports(tmp_path, records=[])

    summary = pd.read_csv(generated.summary_csv)
    high_risk = pd.read_csv(generated.high_risk_csv)
    trends = pd.read_csv(generated.trends_csv)

    assert list(summary["risk_band"]) == ["high", "medium", "low"]
    assert summary["account_count"].sum() == 0
    assert summary["portfolio_proportion"].sum() == 0
    assert high_risk.empty
    assert trends.empty
    assert generated.trend_status == "no_observed_scoring_periods"
