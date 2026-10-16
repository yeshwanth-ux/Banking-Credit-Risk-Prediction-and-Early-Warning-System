from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_preparation import build_preprocessor, create_split_assignments


def classification_data(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [f"R-{index:03d}" for index in range(rows)],
            "target": [index % 2 for index in range(rows)],
            "amount": np.arange(rows, dtype=float),
            "category": ["A" if index % 3 else "B" for index in range(rows)],
        }
    )


def test_stratified_split_is_reproducible_and_balanced() -> None:
    data = classification_data()

    first, definition = create_split_assignments(
        data, target_column="target", id_column="record_id", random_state=42
    )
    second, _ = create_split_assignments(
        data, target_column="target", id_column="record_id", random_state=42
    )

    pd.testing.assert_series_equal(first, second)
    assert definition.method == "stratified_random"
    assert definition.counts == {"train": 60, "validation": 20, "test": 20}
    assert definition.target_rates == {"train": 0.5, "validation": 0.5, "test": 0.5}


def test_reliable_dates_select_chronological_split() -> None:
    data = classification_data(10)
    data["prediction_date"] = pd.date_range("2026-01-01", periods=10)

    assignments, definition = create_split_assignments(
        data,
        target_column="target",
        id_column="record_id",
        date_column="prediction_date",
    )

    assert assignments.tolist() == ["train"] * 6 + ["validation"] * 2 + ["test"] * 2
    assert definition.method == "time_based"
    assert definition.random_state is None
    assert definition.train_end == "2026-01-06"
    assert definition.validation_end == "2026-01-08"


def test_unreliable_dates_fall_back_to_stratified_split() -> None:
    data = classification_data()
    data["prediction_date"] = pd.NaT

    _, definition = create_split_assignments(
        data,
        target_column="target",
        id_column="record_id",
        date_column="prediction_date",
    )

    assert definition.method == "stratified_random"
    assert definition.date_column is None


def test_split_rejects_duplicate_identifiers() -> None:
    data = classification_data()
    data.loc[1, "record_id"] = data.loc[0, "record_id"]

    with pytest.raises(ValueError, match="unique"):
        create_split_assignments(data, target_column="target", id_column="record_id")


def test_preprocessor_is_fit_only_on_training_data() -> None:
    training = pd.DataFrame(
        {"amount": [10.0, 20.0, np.nan], "category": ["A", "B", None]}
    )
    validation = pd.DataFrame({"amount": [1000.0], "category": ["UNSEEN"]})
    preprocessor = build_preprocessor(["amount"], ["category"])

    preprocessor.fit(training)
    transformed = preprocessor.transform(validation)
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]

    assert scaler.mean_[0] == pytest.approx(15.0)
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()
