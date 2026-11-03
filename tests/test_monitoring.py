"""Tests for privacy-safe model and operational monitoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from src.monitoring import ApplicationMonitor, build_training_profile, feature_drift


def training_rows(count: int = 100) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "duration_months": 12 + index % 36,
                "credit_amount_dm": 1000 + index * 25,
                "age_years": 21 + index % 50,
                "credit_amount_per_month": 50 + index,
                "checking_account_status": 1 + index % 4,
                "credit_history": index % 5,
                "purpose": index % 11,
                "savings": 1 + index % 5,
                "employment_duration": 1 + index % 5,
                "installment_rate": 1 + index % 4,
                "personal_status_sex": 1 + index % 4,
                "other_debtors": 1 + index % 3,
                "present_residence": 1 + index % 4,
                "property": 1 + index % 4,
                "other_installment_plans": 1 + index % 3,
                "housing": 1 + index % 3,
                "number_credits": 1 + index % 4,
                "job": 1 + index % 4,
                "people_liable": 1 + index % 2,
                "telephone": 1 + index % 2,
                "foreign_worker": 1 + index % 2,
                "young_borrower_flag": index % 2,
                "long_duration_flag": index % 2,
            }
        )
    return pd.DataFrame(rows)


def test_training_profile_contains_only_aggregate_distributions() -> None:
    profile = build_training_profile(training_rows())

    assert profile["source"] == "training split only"
    assert profile["row_count"] == 100
    assert len(profile["features"]) == 23
    assert "records" not in profile
    assert len(profile["features"]["age_years"]["proportions"]) == 10


def test_drift_flags_shifted_numeric_and_categorical_inputs() -> None:
    training = training_rows()
    profile = build_training_profile(training)
    shifted = training.tail(40).copy()
    shifted["age_years"] = 120
    shifted["housing"] = 99

    results = feature_drift(shifted, profile, minimum_samples=30)
    by_feature = {item["feature"]: item for item in results}

    assert by_feature["age_years"]["method"] == "psi"
    assert by_feature["age_years"]["status"] == "alert"
    assert by_feature["housing"]["method"] == "tvd"
    assert by_feature["housing"]["status"] == "alert"


def test_monitor_summarizes_operations_risk_distribution_and_scoring_age() -> None:
    training = training_rows()
    monitor = ApplicationMonitor(build_training_profile(training), minimum_drift_samples=2)
    scored_at = datetime(2026, 11, 3, 12, tzinfo=UTC)
    monitor.record_request("/predict", 200, 20.0)
    monitor.record_request("/predict", 500, 40.0)
    monitor.record_prediction(training.iloc[0].to_dict(), "low", scored_at)
    monitor.record_prediction(training.iloc[1].to_dict(), "high", scored_at)

    summary = monitor.summary("model-7", now=scored_at + timedelta(minutes=5))

    assert summary["request_count"] == 2
    assert summary["error_count"] == 1
    assert summary["error_rate"] == 0.5
    assert summary["average_latency_ms"] == 30.0
    assert summary["risk_counts"] == {"high": 1, "medium": 0, "low": 1}
    assert summary["risk_proportions"] == {"high": 0.5, "medium": 0.0, "low": 0.5}
    assert summary["model_version"] == "model-7"
    assert summary["last_scoring_age_seconds"] == 300
