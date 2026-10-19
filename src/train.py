"""Train and compare class-weighted baseline credit-risk models."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_preparation import (
    DEFAULT_DATA_DIR,
    PROTOTYPE_CATEGORICAL_FEATURES,
    PROTOTYPE_NUMERIC_FEATURES,
    build_preprocessor,
)
from src.evaluate import BinaryModelMetrics, evaluate_binary_predictions

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "reports" / "baseline_model_results.json"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "reports" / "baseline_model_results.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "baseline_model_results.md"
DEFAULT_CONFUSION_FIGURE = PROJECT_ROOT / "reports" / "figures" / "baseline_confusion_matrices.png"
TARGET_COLUMN = "bad_credit"
ID_COLUMN = "source_record_id"
DECISION_THRESHOLD = 0.50


@dataclass(frozen=True)
class BaselineRun:
    models: dict[str, Pipeline]
    metrics: dict[str, BinaryModelMetrics]
    training_rows: int
    validation_rows: int
    training_bad_credit_rate: float
    validation_bad_credit_rate: float
    transformed_feature_count: int


def _load_split(path: Path, expected_split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared {expected_split} dataset is missing: {path}")
    data = pd.read_csv(path)
    required = {
        ID_COLUMN,
        "split",
        TARGET_COLUMN,
        *PROTOTYPE_NUMERIC_FEATURES,
        *PROTOTYPE_CATEGORICAL_FEATURES,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{expected_split} dataset is missing columns: {', '.join(missing)}")
    if set(data["split"].unique()) != {expected_split}:
        raise ValueError(f"Dataset does not contain only the {expected_split} split")
    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError(f"{expected_split} identifiers must be populated and unique")
    if set(data[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError(f"{expected_split} target must contain both binary classes")
    return data


def build_baseline_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return deterministic class-weighted baseline pipelines."""

    preprocessor = build_preprocessor(
        PROTOTYPE_NUMERIC_FEATURES,
        PROTOTYPE_CATEGORICAL_FEATURES,
    )
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate_baselines(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    *,
    threshold: float = DECISION_THRESHOLD,
) -> BaselineRun:
    """Fit on training rows and evaluate once on validation rows."""

    overlap = set(training_data[ID_COLUMN]) & set(validation_data[ID_COLUMN])
    if overlap:
        raise ValueError("Training and validation identifiers overlap")
    feature_columns = PROTOTYPE_NUMERIC_FEATURES + PROTOTYPE_CATEGORICAL_FEATURES
    train_features = training_data[feature_columns]
    train_target = training_data[TARGET_COLUMN].to_numpy()
    validation_features = validation_data[feature_columns]
    validation_target = validation_data[TARGET_COLUMN].to_numpy()

    models = build_baseline_models()
    metrics: dict[str, BinaryModelMetrics] = {}
    for name, model in models.items():
        model.fit(train_features, train_target)
        probabilities = model.predict_proba(validation_features)[:, 1]
        metrics[name] = evaluate_binary_predictions(
            validation_target,
            probabilities,
            threshold=threshold,
        )
    transformed_count = len(
        models["logistic_regression"].named_steps["preprocessor"].get_feature_names_out()
    )
    return BaselineRun(
        models=models,
        metrics=metrics,
        training_rows=len(training_data),
        validation_rows=len(validation_data),
        training_bad_credit_rate=float(training_data[TARGET_COLUMN].mean()),
        validation_bad_credit_rate=float(validation_data[TARGET_COLUMN].mean()),
        transformed_feature_count=transformed_count,
    )


def _results_payload(run: BaselineRun) -> dict[str, Any]:
    return {
        "experiment": "day_8_baseline_models",
        "dataset": "South German Credit research prototype",
        "target": TARGET_COLUMN,
        "positive_class": "bad credit",
        "evaluation_split": "validation",
        "test_split_used": False,
        "decision_threshold": DECISION_THRESHOLD,
        "class_imbalance_strategy": {
            "logistic_regression": "class_weight=balanced",
            "random_forest": "class_weight=balanced_subsample",
        },
        "training_rows": run.training_rows,
        "validation_rows": run.validation_rows,
        "training_bad_credit_rate": run.training_bad_credit_rate,
        "validation_bad_credit_rate": run.validation_bad_credit_rate,
        "transformed_feature_count": run.transformed_feature_count,
        "models": {name: metrics.to_dict() for name, metrics in run.metrics.items()},
    }


def _write_csv(path: Path, metrics: dict[str, BinaryModelMetrics]) -> None:
    rows = [{"model": name, **values.to_dict()} for name, values in metrics.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_figure(path: Path, metrics: dict[str, BinaryModelMetrics]) -> None:
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    axes_array = np.atleast_1d(axes)
    for axis, (name, values) in zip(axes_array, metrics.items(), strict=True):
        matrix = np.array(
            [
                [values.true_negatives, values.false_positives],
                [values.false_negatives, values.true_positives],
            ]
        )
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks([0, 1], ["Good", "Bad"])
        axis.set_yticks([0, 1], ["Good", "Bad"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.suptitle("Validation confusion matrices at threshold 0.50")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_markdown(path: Path, run: BaselineRun) -> None:
    lines = [
        "# Baseline credit risk model results",
        "",
        "Both models use class weights and a fixed 0.50 threshold. They were fitted on "
        "the training split and evaluated on the validation split. The test split was "
        "not read or evaluated.",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | False alerts | Missed defaults |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in run.metrics.items():
        lines.append(
            f"| {name.replace('_', ' ').title()} | {values.roc_auc:.4f} | "
            f"{values.pr_auc:.4f} | {values.precision:.4f} | {values.recall:.4f} | "
            f"{values.f1_score:.4f} | {values.false_positives} | "
            f"{values.false_negatives} |"
        )
    fewest_missed = min(run.metrics, key=lambda name: run.metrics[name].false_negatives)
    fewest_alerts = min(run.metrics, key=lambda name: run.metrics[name].false_positives)
    lines.extend(
        [
            "",
            "## Operational trade-off",
            "",
            f"`{fewest_missed}` misses the fewest bad-credit cases at this threshold; "
            "higher recall reduces missed defaults but can send more good-credit cases "
            "to manual review.",
            "",
            f"`{fewest_alerts}` produces the fewest false alerts at this threshold; "
            "fewer false positives reduce review workload but may increase missed "
            "bad-credit cases.",
            "",
            "Model selection and any threshold change belong to validation. Final test "
            "evaluation remains deferred until the model and threshold are frozen.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_baseline_training(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    results_json: Path = DEFAULT_RESULTS_JSON,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    report_path: Path = DEFAULT_REPORT,
    confusion_figure: Path = DEFAULT_CONFUSION_FIGURE,
) -> BaselineRun:
    """Run baseline training without opening the held-out test file."""

    training = _load_split(data_dir / "train.csv", "train")
    validation = _load_split(data_dir / "validation.csv", "validation")
    run = train_and_evaluate_baselines(training, validation)

    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in run.models.items():
        joblib.dump(model, model_dir / f"baseline_{name}.joblib")

    payload = _results_payload(run)
    payload["generated_at_utc"] = datetime.now(tz=UTC).isoformat()
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(results_csv, run.metrics)
    _write_markdown(report_path, run)
    _write_confusion_figure(confusion_figure, run.metrics)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    run = run_baseline_training(data_dir=args.data_dir, model_dir=args.model_dir)
    for name, values in run.metrics.items():
        print(
            f"{name}: ROC-AUC={values.roc_auc:.4f}, PR-AUC={values.pr_auc:.4f}, "
            f"precision={values.precision:.4f}, recall={values.recall:.4f}, "
            f"F1={values.f1_score:.4f}"
        )


if __name__ == "__main__":
    main()
