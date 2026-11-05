"""Tests for immutable final-evaluation helpers without opening the real test split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.final_evaluation import calibration_table

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_table_accounts_for_every_sample() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.25, 0.75, 0.95])

    table = calibration_table(target, probabilities, bins=5)

    assert len(table) == 5
    assert table["sample_count"].sum() == 4
    assert table.loc[0, "mean_predicted_probability"] == pytest.approx(0.05)
    assert table.loc[4, "observed_bad_credit_rate"] == 1.0


def test_calibration_table_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        calibration_table(np.array([0, 1]), np.array([0.2, 1.1]))


def test_committed_final_evaluation_is_complete_and_immutable() -> None:
    result = json.loads(
        (ROOT / "reports" / "final_test_evaluation.json").read_text(encoding="utf-8")
    )

    assert result["evaluation_status"] == "completed_once"
    assert result["test_rows"] == 200
    assert result["decision_threshold"] == 0.44
    assert result["test_split_used_for_model_or_threshold_selection"] is False
    assert len(result["test_data_sha256"]) == 64
    assert len(result["model_sha256"]) == 64
    assert result["test_metrics"]["roc_auc"] == pytest.approx(0.8142857143)
    assert result["calibration"]["brier_score"] == pytest.approx(0.1739278449)
