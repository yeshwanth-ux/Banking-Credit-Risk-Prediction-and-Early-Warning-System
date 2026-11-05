"""Evaluate the frozen champion once on the untouched test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data_preparation import DEFAULT_DATA_DIR
from src.evaluate import evaluate_binary_predictions
from src.optimize import expected_calibration_error
from src.predict import DEFAULT_CONFIG, DEFAULT_FROZEN_MODEL, FEATURE_COLUMNS
from src.train import TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DATA = DEFAULT_DATA_DIR / "test.csv"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "final_test_evaluation.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "final_test_calibration.csv"
DEFAULT_OUTPUT_REPORT = PROJECT_ROOT / "reports" / "final_test_evaluation.md"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_table(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Return equal-width calibration bins, including empty bins."""

    actual = np.asarray(target, dtype="int64")
    scores = np.asarray(probabilities, dtype="float64")
    if len(actual) == 0 or len(actual) != len(scores):
        raise ValueError("Calibration inputs must be non-empty and equal length")
    if bins < 2:
        raise ValueError("Calibration requires at least two bins")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between zero and one")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(scores, boundaries[1:-1], right=True), 0, bins - 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        selected = assignments == index
        rows.append(
            {
                "bin": index + 1,
                "lower_bound": float(boundaries[index]),
                "upper_bound": float(boundaries[index + 1]),
                "sample_count": int(selected.sum()),
                "mean_predicted_probability": (
                    float(scores[selected].mean()) if selected.any() else None
                ),
                "observed_bad_credit_rate": (
                    float(actual[selected].mean()) if selected.any() else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _render_report(result: dict[str, Any]) -> str:
    metrics = result["test_metrics"]
    calibration = result["calibration"]
    return f"""# Final untouched-test evaluation

**Evaluation status:** completed once<br>
**Evaluated at:** {result['evaluated_at_utc']}<br>
**Dataset:** {result['dataset']}<br>
**Target:** `{result['target']}`<br>
**Test rows:** {result['test_rows']}<br>
**Decision threshold:** {result['decision_threshold']:.2f} (selected on validation data)<br>
**Model version:** {result['model_version']}

## Test metrics

| Metric | Value |
|---|---:|
| ROC-AUC | {metrics['roc_auc']:.4f} |
| PR-AUC | {metrics['pr_auc']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 | {metrics['f1_score']:.4f} |
| Brier score | {calibration['brier_score']:.4f} |
| Expected calibration error | {calibration['expected_calibration_error']:.4f} |

## Confusion matrix

| | Predicted good | Predicted bad |
|---|---:|---:|
| Actual good | {metrics['true_negatives']} | {metrics['false_positives']} |
| Actual bad | {metrics['false_negatives']} | {metrics['true_positives']} |

## Interpretation

This is the only final evaluation on the held-out test split. The threshold was
frozen from validation analysis before test access and was not tuned on these
results. The dataset models historical contract non-compliance, not a current
90-day probability of default. The small, oversampled 1973-1975 dataset cannot
establish production performance, calibration, fairness, or regulatory fitness.
"""


def evaluate_final_test(
    *,
    test_path: Path = DEFAULT_TEST_DATA,
    model_path: Path = DEFAULT_FROZEN_MODEL,
    config_path: Path = DEFAULT_CONFIG,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_report: Path = DEFAULT_OUTPUT_REPORT,
) -> dict[str, Any]:
    """Evaluate once and refuse to overwrite the immutable final result."""

    existing = [path for path in (output_json, output_csv, output_report) if path.exists()]
    if existing:
        raise FileExistsError(
            "Final test evaluation already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing)
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_hash = file_sha256(model_path)
    if model_hash != config["model"]["sha256"]:
        raise ValueError("Frozen model checksum does not match the classification config")
    test = pd.read_csv(test_path)
    required = {"source_record_id", "split", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required - set(test.columns))
    if missing:
        raise ValueError("Test dataset is missing columns: " + ", ".join(missing))
    if set(test["split"]) != {"test"}:
        raise ValueError("Final evaluation input must contain only the untouched test split")
    model = joblib.load(model_path)
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    target = test[TARGET_COLUMN].to_numpy(dtype="int64")
    threshold = float(config["risk_bands"]["medium_threshold"])
    metrics = evaluate_binary_predictions(target, probabilities, threshold=threshold)
    calibration = calibration_table(target, probabilities)
    result = {
        "evaluation_version": "1.0.0",
        "evaluation_status": "completed_once",
        "evaluated_at_utc": datetime.now(tz=UTC).isoformat(),
        "dataset": "South German Credit research prototype",
        "target": TARGET_COLUMN,
        "test_rows": len(test),
        "test_bad_credit_rate": float(target.mean()),
        "test_data_sha256": file_sha256(test_path),
        "model_version": config["model"]["version"],
        "model_sha256": model_hash,
        "configuration_version": config["configuration_version"],
        "decision_threshold": threshold,
        "threshold_source": "validation cost analysis; not selected on test data",
        "test_metrics": metrics.to_dict(),
        "calibration": {
            "brier_score": float(brier_score_loss(target, probabilities)),
            "expected_calibration_error": expected_calibration_error(
                target,
                probabilities,
            ),
            "bins": 10,
            "calibration_table": output_csv.relative_to(PROJECT_ROOT).as_posix(),
        },
        "test_split_used_for_model_or_threshold_selection": False,
    }
    _atomic_text(output_json, json.dumps(result, indent=2) + "\n")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(output_csv, index=False)
    _atomic_text(output_report, _render_report(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_FROZEN_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    args = parser.parse_args()
    result = evaluate_final_test(
        test_path=args.test,
        model_path=args.model,
        config_path=args.config,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_report=args.output_report,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
