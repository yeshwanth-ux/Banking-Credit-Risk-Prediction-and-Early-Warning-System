from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.data_preparation import (
    PROTOTYPE_CATEGORICAL_FEATURES,
    PROTOTYPE_NUMERIC_FEATURES,
)
from src.optimize import (
    build_searches,
    expected_calibration_error,
    optimize_and_evaluate,
    run_optimization,
    training_reference_values,
)


def model_frame(split: str, rows: int, id_offset: int) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "source_record_id": [f"OPT-{id_offset + index:04d}" for index in range(rows)],
        "split": [split] * rows,
        "bad_credit": [index % 2 for index in range(rows)],
    }
    for position, column in enumerate(PROTOTYPE_NUMERIC_FEATURES):
        data[column] = [float(index + position + (index % 2) * 5) for index in range(rows)]
    for position, column in enumerate(PROTOTYPE_CATEGORICAL_FEATURES):
        data[column] = [int((index + position) % 3) for index in range(rows)]
    return pd.DataFrame(data)


def test_expected_calibration_error_for_perfect_and_miscalibrated_scores() -> None:
    target = np.array([0, 0, 1, 1])

    assert expected_calibration_error(target, np.array([0.0, 0.0, 1.0, 1.0])) == 0
    assert expected_calibration_error(target, np.array([0.5, 0.5, 0.5, 0.5])) == 0
    assert expected_calibration_error(target, np.array([1.0, 1.0, 0.0, 0.0])) == 1


def test_searches_use_stratified_cross_validation_and_average_precision() -> None:
    searches = build_searches(iterations=2, cv_splits=3)

    assert set(searches) == {"logistic_regression", "random_forest"}
    for search in searches.values():
        assert isinstance(search.cv, StratifiedKFold)
        assert search.cv.n_splits == 3
        assert search.scoring == "average_precision"
        assert search.n_iter == 2


def test_training_references_use_medians_and_modes() -> None:
    data = model_frame("train", 7, 0)

    references = training_reference_values(data)

    assert references[PROTOTYPE_NUMERIC_FEATURES[0]] == 6.0
    assert references[PROTOTYPE_CATEGORICAL_FEATURES[0]] == 0


def test_small_optimization_generates_importance_and_explanations() -> None:
    training = model_frame("train", 60, 0)
    validation = model_frame("validation", 30, 1000)

    run = optimize_and_evaluate(
        training,
        validation,
        iterations=1,
        cv_splits=2,
        permutation_repeats=2,
    )

    assert set(run.results) == {"logistic_regression", "random_forest"}
    assert run.champion in run.results
    assert len(run.permutation_importance[run.champion]) == 23
    assert len(run.explanations) == 3
    assert len(run.explanations[0]["top_contributions"]) == 8
    assert all(0 <= item["predicted_probability"] <= 1 for item in run.explanations)


def test_optimization_run_does_not_require_test_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    reports = tmp_path / "reports"
    data_dir.mkdir()
    reports.mkdir()
    model_frame("train", 60, 0).to_csv(data_dir / "train.csv", index=False)
    model_frame("validation", 30, 1000).to_csv(data_dir / "validation.csv", index=False)
    baseline = {
        "test_split_used": False,
        "models": {
            "logistic_regression": {"roc_auc": 0.5, "pr_auc": 0.5},
            "random_forest": {"roc_auc": 0.5, "pr_auc": 0.5},
        },
    }
    baseline_path = reports / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    run = run_optimization(
        data_dir=data_dir,
        model_dir=model_dir,
        baseline_results_path=baseline_path,
        results_json=reports / "optimized.json",
        results_csv=reports / "optimized.csv",
        importance_csv=reports / "importance.csv",
        explanations_json=reports / "explanations.json",
        report_path=reports / "optimized.md",
        calibration_figure=reports / "calibration.png",
        importance_figure=reports / "importance.png",
        iterations=1,
        cv_splits=2,
        permutation_repeats=1,
    )

    assert run.champion in run.models
    assert (model_dir / "optimized_champion.joblib").is_file()
    payload = json.loads((reports / "optimized.json").read_text(encoding="utf-8"))
    assert payload["test_split_used"] is False
    assert payload["evaluation_split"] == "validation"
    assert (reports / "explanations.json").is_file()
    assert (reports / "calibration.png").is_file()


def test_calibration_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        expected_calibration_error(np.array([0, 1]), np.array([0.2, 1.2]))
