from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_preparation import (
    PROTOTYPE_CATEGORICAL_FEATURES,
    PROTOTYPE_NUMERIC_FEATURES,
)
from src.train import build_baseline_models, run_baseline_training


def model_frame(split: str, rows: int, id_offset: int) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "source_record_id": [f"R-{id_offset + index:04d}" for index in range(rows)],
        "split": [split] * rows,
        "bad_credit": [index % 2 for index in range(rows)],
    }
    for position, column in enumerate(PROTOTYPE_NUMERIC_FEATURES):
        data[column] = [float(index + position + (index % 2) * 5) for index in range(rows)]
    for position, column in enumerate(PROTOTYPE_CATEGORICAL_FEATURES):
        data[column] = [int((index + position) % 3) for index in range(rows)]
    return pd.DataFrame(data)


def test_baseline_models_use_class_weights() -> None:
    models = build_baseline_models()

    logistic = models["logistic_regression"].named_steps["classifier"]
    forest = models["random_forest"].named_steps["classifier"]
    assert logistic.class_weight == "balanced"
    assert forest.class_weight == "balanced_subsample"
    assert logistic.random_state == 42
    assert forest.random_state == 42


def test_training_run_writes_structured_results_without_test_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    data_dir.mkdir()
    model_frame("train", 80, 0).to_csv(data_dir / "train.csv", index=False)
    model_frame("validation", 40, 1000).to_csv(data_dir / "validation.csv", index=False)

    run = run_baseline_training(
        data_dir=data_dir,
        model_dir=model_dir,
        results_json=report_dir / "results.json",
        results_csv=report_dir / "results.csv",
        report_path=report_dir / "results.md",
        confusion_figure=report_dir / "confusion.png",
    )

    assert set(run.metrics) == {"logistic_regression", "random_forest"}
    assert run.training_rows == 80
    assert run.validation_rows == 40
    assert run.transformed_feature_count > len(PROTOTYPE_NUMERIC_FEATURES)
    assert (model_dir / "baseline_logistic_regression.joblib").is_file()
    assert (model_dir / "baseline_random_forest.joblib").is_file()
    assert (report_dir / "results.csv").is_file()
    assert (report_dir / "results.md").is_file()
    assert (report_dir / "confusion.png").is_file()

    payload = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["evaluation_split"] == "validation"
    assert payload["test_split_used"] is False
    assert set(payload["models"]) == {"logistic_regression", "random_forest"}
