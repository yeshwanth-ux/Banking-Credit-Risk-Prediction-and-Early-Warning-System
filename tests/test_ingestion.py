from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingestion import COLUMNS, read_source, validate_financial_fields, validate_records


def valid_record() -> dict[str, int]:
    return {
        "checking_account_status": 1,
        "duration_months": 12,
        "credit_history": 2,
        "purpose": 3,
        "credit_amount_dm": 2000,
        "savings": 2,
        "employment_duration": 3,
        "installment_rate": 4,
        "personal_status_sex": 2,
        "other_debtors": 1,
        "present_residence": 3,
        "property": 2,
        "age_years": 35,
        "other_installment_plans": 3,
        "housing": 3,
        "number_credits": 1,
        "job": 3,
        "people_liable": 2,
        "telephone": 2,
        "foreign_worker": 1,
        "credit_risk": 1,
        "bad_credit": 0,
    }


def test_read_source_standardizes_column_names(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    pd.DataFrame({" Loan Amount ": [100], "Customer-ID": ["A-1"]}).to_csv(path, index=False)

    result = read_source(path)

    assert list(result.columns) == ["loan_amount", "customer_id"]


def test_accepts_valid_record_and_creates_stable_identifier() -> None:
    first = validate_records(pd.DataFrame([valid_record()], columns=COLUMNS))
    second = validate_records(pd.DataFrame([valid_record()], columns=COLUMNS))

    assert not first.rejected
    assert len(first.accepted) == 1
    assert len(first.accepted[0]["source_record_id"]) == 64
    assert first.accepted[0]["source_record_id"] == second.accepted[0]["source_record_id"]


def test_rejects_missing_invalid_and_duplicate_records() -> None:
    invalid = valid_record()
    invalid["interest_placeholder"] = 1
    invalid.pop("duration_months")
    with pytest.raises(ValueError, match="data dictionary"):
        validate_records(pd.DataFrame([invalid]))

    duplicate = valid_record()
    duplicate["credit_amount_dm"] = -1
    result = validate_records(pd.DataFrame([duplicate, duplicate], columns=COLUMNS))

    assert len(result.rejected) == 2
    assert "invalid_credit_amount_dm" in result.rejected[0].errors
    assert "duplicate_record" in result.rejected[1].errors


def test_rejects_target_mismatch_and_invalid_category() -> None:
    record = valid_record()
    record["bad_credit"] = 1
    record["housing"] = 99

    result = validate_records(pd.DataFrame([record], columns=COLUMNS))

    assert result.rejected[0].errors == ("invalid_category_housing", "target_mismatch")


def test_financial_rules_cover_amount_rate_and_repayment_dates() -> None:
    errors = validate_financial_fields(
        {
            "principal_amount": "-1",
            "paid_amount": "not-a-number",
            "interest_rate_pct": "101",
            "application_date": "2026-01-01",
            "origination_date": "2026-02-01",
            "due_date": "2026-01-15",
            "payment_date": "bad-date",
        }
    )

    assert errors == [
        "invalid_principal_amount",
        "invalid_paid_amount",
        "invalid_interest_rate_pct",
        "invalid_payment_date",
        "due_date_before_origination",
    ]
