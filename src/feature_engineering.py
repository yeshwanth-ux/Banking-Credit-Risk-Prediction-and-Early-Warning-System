"""Reusable point-in-time credit-risk feature engineering."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.database import DatabaseSettings, connection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OPERATIONAL_REQUIRED_COLUMNS = {
    "principal_amount",
    "interest_rate_pct",
    "term_months",
    "annual_income",
    "scheduled_amount_due",
    "paid_amount_as_of_date",
    "outstanding_amount_as_of_date",
    "installments_due",
    "installments_overdue",
    "customer_outstanding_debt",
    "customer_monthly_principal_obligation",
    "customer_loan_count",
    "previous_default_count",
    "account_activity_count_90d",
    "account_net_amount_90d",
}

OPERATIONAL_SOURCE_QUERY = """
SELECT
    t.*,
    COALESCE(history.customer_loan_count, 0)::INTEGER AS customer_loan_count,
    COALESCE(history.previous_default_count, 0)::INTEGER AS previous_default_count,
    COALESCE(history.customer_total_principal, 0)::NUMERIC(18, 2)
        AS customer_total_principal,
    COALESCE(history.customer_outstanding_debt, 0)::NUMERIC(18, 2)
        AS customer_outstanding_debt,
    COALESCE(history.customer_monthly_principal_obligation, 0)::NUMERIC(18, 2)
        AS customer_monthly_principal_obligation,
    COALESCE(delay.average_days_past_due, 0)::NUMERIC(10, 4)
        AS average_days_past_due,
    COALESCE(delay.late_payment_count, 0)::INTEGER AS late_payment_count,
    COALESCE(activity.account_activity_count, 0)::INTEGER AS account_activity_count,
    COALESCE(activity.account_activity_count_90d, 0)::INTEGER
        AS account_activity_count_90d,
    COALESCE(activity.account_net_amount_90d, 0)::NUMERIC(18, 2)
        AS account_net_amount_90d,
    activity.latest_account_balance,
    NULL::NUMERIC AS credit_limit
FROM trusted_loan_analytical_dataset t
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS customer_loan_count,
        COUNT(*) FILTER (
            WHERE EXISTS (
                SELECT 1
                FROM repayment_history default_rh
                WHERE default_rh.loan_id = customer_loan.loan_id
                  AND default_rh.due_date <= t.prediction_date - 90
                  AND (
                      default_rh.payment_date IS NULL
                      OR default_rh.payment_date > default_rh.due_date + 90
                  )
            )
        ) AS previous_default_count,
        SUM(customer_loan.principal_amount) AS customer_total_principal,
        SUM(
            GREATEST(
                customer_loan.principal_amount - COALESCE(customer_paid.paid_amount, 0),
                0
            )
        ) AS customer_outstanding_debt,
        SUM(customer_loan.principal_amount / customer_loan.term_months)
            AS customer_monthly_principal_obligation
    FROM loans customer_loan
    LEFT JOIN LATERAL (
        SELECT SUM(rh.paid_amount) AS paid_amount
        FROM repayment_history rh
        WHERE rh.loan_id = customer_loan.loan_id
          AND rh.payment_date <= t.prediction_date
    ) customer_paid ON TRUE
    WHERE customer_loan.customer_id = t.customer_id
      AND customer_loan.application_date <= t.prediction_date
) history ON TRUE
LEFT JOIN LATERAL (
    SELECT
        AVG(
            CASE
                WHEN rh.payment_date <= t.prediction_date
                    THEN GREATEST(rh.payment_date - rh.due_date, 0)
                ELSE GREATEST(t.prediction_date - rh.due_date, 0)
            END
        ) AS average_days_past_due,
        COUNT(*) FILTER (
            WHERE CASE
                WHEN rh.payment_date <= t.prediction_date
                    THEN rh.payment_date - rh.due_date > 0
                ELSE t.prediction_date - rh.due_date > 0
            END
        ) AS late_payment_count
    FROM repayment_history rh
    WHERE rh.loan_id = t.loan_id
      AND rh.due_date <= t.prediction_date
) delay ON TRUE
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS account_activity_count,
        COUNT(*) FILTER (
            WHERE aa.activity_timestamp >= t.prediction_date - 89
        ) AS account_activity_count_90d,
        SUM(aa.amount) FILTER (
            WHERE aa.activity_timestamp >= t.prediction_date - 89
        ) AS account_net_amount_90d,
        (ARRAY_AGG(aa.balance_after ORDER BY aa.activity_timestamp DESC)
            FILTER (WHERE aa.balance_after IS NOT NULL))[1] AS latest_account_balance
    FROM account_activity aa
    WHERE aa.customer_id = t.customer_id
      AND aa.activity_timestamp < t.prediction_date + 1
) activity ON TRUE
ORDER BY t.loan_id, t.prediction_date
"""


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("float64")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide finite values by positive denominators and return NaN otherwise."""

    top = _numeric(numerator)
    bottom = _numeric(denominator)
    result = np.full(len(top), np.nan, dtype="float64")
    valid = np.isfinite(top) & np.isfinite(bottom) & (bottom > 0)
    np.divide(top, bottom, out=result, where=valid)
    return pd.Series(result, index=top.index, dtype="float64")


def calculate_monthly_payment(
    principal: pd.Series,
    annual_interest_rate_pct: pd.Series,
    term_months: pd.Series,
) -> pd.Series:
    """Calculate the standard amortizing monthly payment for each loan."""

    amount = _numeric(principal)
    annual_rate = _numeric(annual_interest_rate_pct)
    months = _numeric(term_months)
    monthly_rate = annual_rate / 1200.0
    result = np.full(len(amount), np.nan, dtype="float64")
    valid = (
        np.isfinite(amount)
        & np.isfinite(monthly_rate)
        & np.isfinite(months)
        & (amount > 0)
        & (annual_rate >= 0)
        & (months > 0)
    )
    zero_rate = valid & (monthly_rate == 0)
    np.divide(amount, months, out=result, where=zero_rate)
    interest_bearing = valid & (monthly_rate > 0)
    factor = np.power(1.0 + monthly_rate, months)
    denominator = factor - 1.0
    np.divide(
        amount * monthly_rate * factor,
        denominator,
        out=result,
        where=interest_bearing & np.isfinite(factor) & (denominator > 0),
    )
    return pd.Series(result, index=amount.index, dtype="float64")


def _require_columns(data: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")


def engineer_operational_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create financial, repayment, history, and activity features."""

    _require_columns(data, OPERATIONAL_REQUIRED_COLUMNS)
    features = data.copy()
    features["monthly_income"] = _numeric(features["annual_income"]) / 12.0
    features["estimated_monthly_installment"] = calculate_monthly_payment(
        features["principal_amount"],
        features["interest_rate_pct"],
        features["term_months"],
    )
    features["loan_to_income_ratio"] = safe_ratio(
        features["principal_amount"], features["annual_income"]
    )
    features["loan_installment_burden_ratio"] = safe_ratio(
        features["estimated_monthly_installment"], features["monthly_income"]
    )
    features["debt_to_income_ratio"] = safe_ratio(
        features["customer_monthly_principal_obligation"], features["monthly_income"]
    )
    features["outstanding_debt_to_income_ratio"] = safe_ratio(
        features["customer_outstanding_debt"], features["annual_income"]
    )
    features["loan_outstanding_ratio"] = safe_ratio(
        features["outstanding_amount_as_of_date"], features["principal_amount"]
    )
    features["payment_completion_ratio"] = safe_ratio(
        features["paid_amount_as_of_date"], features["scheduled_amount_due"]
    ).fillna(0.0)
    features["missed_payment_ratio"] = safe_ratio(
        features["installments_overdue"], features["installments_due"]
    ).fillna(0.0)
    features["missed_payment_count"] = (
        _numeric(features["installments_overdue"]).fillna(0).astype("int64")
    )
    features["previous_default_flag"] = (
        _numeric(features["previous_default_count"]).fillna(0).gt(0).astype("int8")
    )
    features["multiple_loan_flag"] = (
        _numeric(features["customer_loan_count"]).fillna(0).gt(1).astype("int8")
    )
    features["has_recent_account_activity"] = (
        _numeric(features["account_activity_count_90d"]).fillna(0).gt(0).astype("int8")
    )
    features["account_net_outflow_90d"] = (
        -_numeric(features["account_net_amount_90d"])
    ).clip(lower=0)

    if {"latest_account_balance", "credit_limit"}.issubset(features.columns):
        features["credit_utilization_ratio"] = safe_ratio(
            features["latest_account_balance"].clip(lower=0),
            features["credit_limit"],
        )
        features["credit_utilization_available"] = (
            features["credit_utilization_ratio"].notna().astype("int8")
        )
    else:
        features["credit_utilization_ratio"] = np.nan
        features["credit_utilization_available"] = np.int8(0)
    return features


def engineer_prototype_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create only defensible features from the undated research dataset."""

    required = {"credit_amount_dm", "duration_months", "age_years"}
    _require_columns(data, required)
    features = data.copy()
    features["credit_amount_per_month"] = safe_ratio(
        features["credit_amount_dm"], features["duration_months"]
    )
    features["young_borrower_flag"] = _numeric(features["age_years"]).lt(25).astype("int8")
    features["long_duration_flag"] = (
        _numeric(features["duration_months"]).gt(36).astype("int8")
    )
    return features


def _query_frame(statement: str, settings: DatabaseSettings | None = None) -> pd.DataFrame:
    with connection(settings) as db:
        rows = db.execute(statement).fetchall()
    return pd.DataFrame(rows)


def load_operational_feature_source(
    settings: DatabaseSettings | None = None,
) -> pd.DataFrame:
    return _query_frame(OPERATIONAL_SOURCE_QUERY, settings)


def load_prototype_feature_source(
    settings: DatabaseSettings | None = None,
) -> pd.DataFrame:
    return _query_frame("SELECT * FROM trusted_credit_risk_dataset", settings)


def build_feature_datasets(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    settings: DatabaseSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    operational_source = load_operational_feature_source(settings)
    if operational_source.empty:
        operational_features = operational_source
    else:
        operational_features = engineer_operational_features(operational_source)
    prototype_features = engineer_prototype_features(load_prototype_feature_source(settings))

    output_dir.mkdir(parents=True, exist_ok=True)
    operational_features.to_csv(output_dir / "operational_features.csv", index=False)
    prototype_features.to_csv(output_dir / "prototype_features.csv", index=False)
    return operational_features, prototype_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    operational, prototype = build_feature_datasets(args.output_dir)
    print(f"Operational feature rows: {len(operational):,}")
    print(f"Prototype feature rows: {len(prototype):,}")


if __name__ == "__main__":
    main()
