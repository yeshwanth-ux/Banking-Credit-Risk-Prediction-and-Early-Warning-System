from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    calculate_monthly_payment,
    engineer_operational_features,
    engineer_prototype_features,
    safe_ratio,
)


def operational_record() -> dict[str, object]:
    return {
        "principal_amount": 12000.0,
        "interest_rate_pct": 12.0,
        "term_months": 12,
        "annual_income": 60000.0,
        "scheduled_amount_due": 4000.0,
        "paid_amount_as_of_date": 3000.0,
        "outstanding_amount_as_of_date": 1000.0,
        "installments_due": 4,
        "installments_overdue": 1,
        "customer_outstanding_debt": 18000.0,
        "customer_monthly_principal_obligation": 1500.0,
        "customer_loan_count": 2,
        "previous_default_count": 1,
        "account_activity_count_90d": 3,
        "account_net_amount_90d": -250.0,
        "latest_account_balance": 500.0,
        "credit_limit": 2000.0,
    }


def test_safe_ratio_rejects_zero_negative_and_missing_denominators() -> None:
    result = safe_ratio(
        pd.Series([10, 10, 10, np.nan]),
        pd.Series([2, 0, -1, 2]),
    )

    assert result.iloc[0] == 5.0
    assert result.iloc[1:].isna().all()


def test_monthly_payment_handles_interest_and_zero_rate() -> None:
    result = calculate_monthly_payment(
        pd.Series([12000.0, 12000.0]),
        pd.Series([12.0, 0.0]),
        pd.Series([12, 12]),
    )

    assert result.iloc[0] == pytest.approx(1066.185, rel=1e-5)
    assert result.iloc[1] == pytest.approx(1000.0)


def test_monthly_payment_rejects_invalid_financial_values() -> None:
    result = calculate_monthly_payment(
        pd.Series([0.0, -1.0, 100.0, 100.0]),
        pd.Series([5.0, 5.0, -1.0, 5.0]),
        pd.Series([12, 12, 12, 0]),
    )

    assert result.isna().all()


def test_engineers_financial_burden_features() -> None:
    result = engineer_operational_features(pd.DataFrame([operational_record()])).iloc[0]

    assert result["monthly_income"] == 5000.0
    assert result["loan_to_income_ratio"] == pytest.approx(0.2)
    assert result["loan_installment_burden_ratio"] == pytest.approx(1066.185 / 5000)
    assert result["debt_to_income_ratio"] == pytest.approx(0.3)
    assert result["outstanding_debt_to_income_ratio"] == pytest.approx(0.3)
    assert result["loan_outstanding_ratio"] == pytest.approx(1 / 12)


def test_engineers_repayment_and_history_features() -> None:
    result = engineer_operational_features(pd.DataFrame([operational_record()])).iloc[0]

    assert result["payment_completion_ratio"] == pytest.approx(0.75)
    assert result["missed_payment_ratio"] == pytest.approx(0.25)
    assert result["missed_payment_count"] == 1
    assert result["previous_default_flag"] == 1
    assert result["multiple_loan_flag"] == 1


def test_engineers_account_activity_and_utilization() -> None:
    result = engineer_operational_features(pd.DataFrame([operational_record()])).iloc[0]

    assert result["has_recent_account_activity"] == 1
    assert result["account_net_outflow_90d"] == 250.0
    assert result["credit_utilization_ratio"] == pytest.approx(0.25)
    assert result["credit_utilization_available"] == 1


def test_unavailable_credit_limit_produces_missing_utilization() -> None:
    record = operational_record()
    record["credit_limit"] = None
    result = engineer_operational_features(pd.DataFrame([record])).iloc[0]

    assert math.isnan(result["credit_utilization_ratio"])
    assert result["credit_utilization_available"] == 0


def test_zero_income_does_not_create_infinite_ratios() -> None:
    record = operational_record()
    record["annual_income"] = 0
    result = engineer_operational_features(pd.DataFrame([record])).iloc[0]

    assert math.isnan(result["loan_to_income_ratio"])
    assert math.isnan(result["debt_to_income_ratio"])
    assert math.isnan(result["loan_installment_burden_ratio"])


def test_prototype_features_use_only_available_fields() -> None:
    source = pd.DataFrame(
        [{"credit_amount_dm": 2400, "duration_months": 24, "age_years": 22}]
    )

    result = engineer_prototype_features(source).iloc[0]

    assert result["credit_amount_per_month"] == 100.0
    assert result["young_borrower_flag"] == 1
    assert result["long_duration_flag"] == 0


def test_missing_required_columns_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="Missing required feature columns"):
        engineer_operational_features(pd.DataFrame({"principal_amount": [100]}))
