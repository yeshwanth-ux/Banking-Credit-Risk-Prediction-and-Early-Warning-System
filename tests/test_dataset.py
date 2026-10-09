from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.download_data import validate_dataset


def make_valid_dataset() -> pd.DataFrame:
    columns = [
        "checking_account_status",
        "duration_months",
        "credit_history",
        "purpose",
        "credit_amount_dm",
        "savings",
        "employment_duration",
        "installment_rate",
        "personal_status_sex",
        "other_debtors",
        "present_residence",
        "property",
        "age_years",
        "other_installment_plans",
        "housing",
        "number_credits",
        "job",
        "people_liable",
        "telephone",
        "foreign_worker",
        "credit_risk",
    ]
    data = pd.DataFrame(1, index=range(1000), columns=columns)
    data.loc[0, "credit_risk"] = 0
    data["bad_credit"] = 1 - data["credit_risk"]
    return data


def test_validate_dataset_accepts_expected_schema() -> None:
    validate_dataset(make_valid_dataset())


def test_validate_dataset_rejects_missing_values() -> None:
    data = make_valid_dataset()
    data.loc[0, "age_years"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_dataset(data)


def test_local_raw_dataset_matches_contract() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "banking_data.csv"
    if not dataset_path.exists():
        pytest.skip("Run `python -m src.download_data` to download the ignored raw dataset.")

    validate_dataset(pd.read_csv(dataset_path))
