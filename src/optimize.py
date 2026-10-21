"""Tune, calibrate, compare, and explain credit-risk models."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data_preparation import (
    DEFAULT_DATA_DIR,
    PROTOTYPE_CATEGORICAL_FEATURES,
    PROTOTYPE_NUMERIC_FEATURES,
)
from src.evaluate import BinaryModelMetrics, evaluate_binary_predictions
from src.train import (
    DECISION_THRESHOLD,
    DEFAULT_MODEL_DIR,
    ID_COLUMN,
    TARGET_COLUMN,
    _load_split,
    build_baseline_models,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_RESULTS = PROJECT_ROOT / "reports" / "baseline_model_results.json"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "reports" / "optimized_model_results.json"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "reports" / "optimized_model_results.csv"
DEFAULT_IMPORTANCE_CSV = PROJECT_ROOT / "reports" / "permutation_importance.csv"
DEFAULT_EXPLANATIONS_JSON = PROJECT_ROOT / "reports" / "model_explanations.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "optimized_model_results.md"
DEFAULT_CALIBRATION_FIGURE = PROJECT_ROOT / "reports" / "figures" / "model_calibration.png"
DEFAULT_IMPORTANCE_FIGURE = (
    PROJECT_ROOT / "reports" / "figures" / "permutation_importance.png"
)
FEATURE_COLUMNS = PROTOTYPE_NUMERIC_FEATURES + PROTOTYPE_CATEGORICAL_FEATURES


@dataclass(frozen=True)
class CalibrationMetrics:
    brier_score: float
    expected_calibration_error: float


@dataclass(frozen=True)
class OptimizedModelResult:
    validation_metrics: BinaryModelMetrics
    calibration: CalibrationMetrics
    best_cv_pr_auc: float
    best_parameters: dict[str, Any]


@dataclass(frozen=True)
class OptimizationRun:
    models: dict[str, Pipeline]
    results: dict[str, OptimizedModelResult]
    permutation_importance: dict[str, pd.DataFrame]
    explanations: list[dict[str, Any]]
    champion: str
    cv_strategy: str


def expected_calibration_error(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Calculate equal-width expected calibration error."""

    actual = np.asarray(target, dtype="float64")
    scores = np.asarray(probabilities, dtype="float64")
    if len(actual) == 0 or len(actual) != len(scores):
        raise ValueError("Calibration inputs must be non-empty and equal length")
    if bins < 2:
        raise ValueError("Calibration requires at least two bins")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(scores, boundaries[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            error += float(selected.mean()) * abs(
                float(scores[selected].mean()) - float(actual[selected].mean())
            )
    return error


def build_searches(
    *,
    iterations: int = 8,
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict[str, RandomizedSearchCV]:
    """Create deterministic searches using stratified training-only folds."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    baselines = build_baseline_models(random_state)
    baselines["random_forest"].set_params(classifier__n_jobs=1)
    cross_validation = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    distributions = {
        "logistic_regression": {
            "classifier__C": np.logspace(-3, 2, 20).tolist(),
            "classifier__tol": [1e-5, 1e-4, 1e-3, 1e-2],
        },
        "random_forest": {
            "classifier__n_estimators": [200, 300, 500],
            "classifier__max_depth": [None, 6, 10, 16],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__max_features": ["sqrt", "log2", 0.5],
        },
    }
    return {
        name: RandomizedSearchCV(
            estimator=model,
            param_distributions=distributions[name],
            n_iter=iterations,
            scoring="average_precision",
            cv=cross_validation,
            refit=True,
            random_state=random_state,
            n_jobs=-1,
            return_train_score=False,
        )
        for name, model in baselines.items()
    }


def calculate_permutation_importance(
    model: Pipeline,
    features: pd.DataFrame,
    target: np.ndarray,
    *,
    repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure validation PR-AUC loss after permuting each raw feature."""

    result = permutation_importance(
        model,
        features,
        target,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)


def training_reference_values(training_features: pd.DataFrame) -> dict[str, Any]:
    """Return training-only medians and modes for perturbation explanations."""

    references: dict[str, Any] = {}
    for column in PROTOTYPE_NUMERIC_FEATURES:
        references[column] = float(pd.to_numeric(training_features[column]).median())
    for column in PROTOTYPE_CATEGORICAL_FEATURES:
        mode = training_features[column].mode(dropna=True)
        if mode.empty:
            raise ValueError(f"Cannot explain categorical feature without a mode: {column}")
        references[column] = mode.iloc[0].item() if hasattr(mode.iloc[0], "item") else mode.iloc[0]
    return references


def explain_record(
    model: Pipeline,
    record: pd.DataFrame,
    reference_values: dict[str, Any],
    *,
    top_features: int = 8,
) -> dict[str, Any]:
    """Explain one prediction with model-agnostic single-feature perturbations."""

    if len(record) != 1:
        raise ValueError("Exactly one record is required for a local explanation")
    original_probability = float(model.predict_proba(record[FEATURE_COLUMNS])[:, 1][0])
    reference = pd.DataFrame([reference_values], columns=FEATURE_COLUMNS)
    reference_probability = float(model.predict_proba(reference)[:, 1][0])
    contributions: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        perturbed = record[FEATURE_COLUMNS].copy()
        perturbed.loc[:, feature] = reference_values[feature]
        perturbed_probability = float(model.predict_proba(perturbed)[:, 1][0])
        contributions.append(
            {
                "feature": feature,
                "observed_value": _json_value(record.iloc[0][feature]),
                "reference_value": _json_value(reference_values[feature]),
                "probability_contribution": original_probability - perturbed_probability,
            }
        )
    contributions.sort(key=lambda item: abs(item["probability_contribution"]), reverse=True)
    return {
        "record_id": str(record.iloc[0][ID_COLUMN]),
        "actual_bad_credit": int(record.iloc[0][TARGET_COLUMN]),
        "predicted_probability": original_probability,
        "reference_probability": reference_probability,
        "top_contributions": contributions[:top_features],
    }


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def optimize_and_evaluate(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    *,
    iterations: int = 8,
    cv_splits: int = 5,
    permutation_repeats: int = 20,
) -> OptimizationRun:
    """Tune on training folds, then compare once on validation data."""

    overlap = set(training_data[ID_COLUMN]) & set(validation_data[ID_COLUMN])
    if overlap:
        raise ValueError("Training and validation identifiers overlap")
    train_features = training_data[FEATURE_COLUMNS]
    train_target = training_data[TARGET_COLUMN].to_numpy()
    validation_features = validation_data[FEATURE_COLUMNS]
    validation_target = validation_data[TARGET_COLUMN].to_numpy()
    searches = build_searches(iterations=iterations, cv_splits=cv_splits)
    models: dict[str, Pipeline] = {}
    results: dict[str, OptimizedModelResult] = {}
    importances: dict[str, pd.DataFrame] = {}
    probabilities: dict[str, np.ndarray] = {}

    for name, search in searches.items():
        search.fit(train_features, train_target)
        model = search.best_estimator_
        scores = model.predict_proba(validation_features)[:, 1]
        models[name] = model
        probabilities[name] = scores
        results[name] = OptimizedModelResult(
            validation_metrics=evaluate_binary_predictions(
                validation_target, scores, threshold=DECISION_THRESHOLD
            ),
            calibration=CalibrationMetrics(
                brier_score=float(brier_score_loss(validation_target, scores)),
                expected_calibration_error=expected_calibration_error(
                    validation_target, scores
                ),
            ),
            best_cv_pr_auc=float(search.best_score_),
            best_parameters={key: _json_value(value) for key, value in search.best_params_.items()},
        )
        importances[name] = calculate_permutation_importance(
            model,
            validation_features,
            validation_target,
            repeats=permutation_repeats,
        )

    champion = max(results, key=lambda name: results[name].validation_metrics.pr_auc)
    references = training_reference_values(train_features)
    ranked_indices = np.argsort(probabilities[champion])[::-1][:3]
    explanations = [
        explain_record(models[champion], validation_data.iloc[[index]], references)
        for index in ranked_indices
    ]
    return OptimizationRun(
        models=models,
        results=results,
        permutation_importance=importances,
        explanations=explanations,
        champion=champion,
        cv_strategy=f"StratifiedKFold(n_splits={cv_splits}, shuffle=True, random_state=42)",
    )


def _read_baseline_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline result file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_split_used") is not False or "models" not in payload:
        raise ValueError("Baseline results do not have the expected validation-only structure")
    return payload


def _write_results_csv(path: Path, run: OptimizationRun) -> None:
    rows: list[dict[str, Any]] = []
    for name, result in run.results.items():
        rows.append(
            {
                "model": name,
                "best_cv_pr_auc": result.best_cv_pr_auc,
                **result.validation_metrics.to_dict(),
                **asdict(result.calibration),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_importance_csv(path: Path, run: OptimizationRun) -> None:
    frames: list[pd.DataFrame] = []
    for name, importance in run.permutation_importance.items():
        frame = importance.copy()
        frame.insert(0, "model", name)
        frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _write_calibration_figure(
    path: Path,
    models: dict[str, Pipeline],
    validation_features: pd.DataFrame,
    validation_target: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    for name, model in models.items():
        scores = model.predict_proba(validation_features)[:, 1]
        observed, predicted = calibration_curve(
            validation_target,
            scores,
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", label=name.replace("_", " ").title())
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed bad-credit rate")
    axis.set_title("Validation probability calibration")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_importance_figure(path: Path, run: OptimizationRun) -> None:
    importance = run.permutation_importance[run.champion].head(15).sort_values(
        "importance_mean"
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        importance["feature"],
        importance["importance_mean"],
        xerr=importance["importance_std"],
        color="#355C7D",
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Decrease in validation average precision after permutation")
    axis.set_title(
        f"Permutation importance: {run.champion.replace('_', ' ').title()}"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_markdown(
    path: Path,
    run: OptimizationRun,
    baseline: dict[str, Any],
) -> None:
    lines = [
        "# Optimized model results and explainability",
        "",
        f"**Cross-validation:** `{run.cv_strategy}`",
        "**Search objective:** training-fold average precision (PR-AUC)",
        "**Final comparison split:** validation",
        "**Held-out test used:** no",
        f"**Selected validation champion:** `{run.champion}`",
        "",
        "## Optimized model comparison",
        "",
        "| Model | CV PR-AUC | Validation ROC-AUC | Validation PR-AUC | "
        "Precision | Recall | F1 | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in run.results.items():
        metrics = result.validation_metrics
        lines.append(
            f"| {name.replace('_', ' ').title()} | {result.best_cv_pr_auc:.4f} | "
            f"{metrics.roc_auc:.4f} | {metrics.pr_auc:.4f} | {metrics.precision:.4f} | "
            f"{metrics.recall:.4f} | {metrics.f1_score:.4f} | "
            f"{result.calibration.brier_score:.4f} | "
            f"{result.calibration.expected_calibration_error:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Change from baseline",
            "",
            "| Model | ROC-AUC change | PR-AUC change |",
            "|---|---:|---:|",
        ]
    )
    for name, result in run.results.items():
        previous = baseline["models"][name]
        metrics = result.validation_metrics
        lines.append(
            f"| {name.replace('_', ' ').title()} | "
            f"{metrics.roc_auc - previous['roc_auc']:+.4f} | "
            f"{metrics.pr_auc - previous['pr_auc']:+.4f} |"
        )

    champion_importance = run.permutation_importance[run.champion].head(10)
    lines.extend(
        [
            "",
            "## Primary predictive features",
            "",
            "Permutation importance measures the validation PR-AUC decrease after one "
            "raw feature is shuffled. Larger positive values indicate greater model "
            "reliance on that feature.",
            "",
            "| Feature | Mean importance | Standard deviation |",
            "|---|---:|---:|",
        ]
    )
    for row in champion_importance.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.importance_mean:.4f} | {row.importance_std:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Explanation method and limitations",
            "",
            "Local explanations replace one raw feature at a time with its training-set "
            "median or mode and measure the change in predicted probability. They show "
            "model sensitivity for the selected record, not causality, approval reasons, "
            "or additive probability decompositions.",
            "",
            "Permutation results can be unstable when features are correlated because "
            "related variables can substitute for one another. Calibration estimates use "
            "only 200 validation records and require confirmation on later data. The "
            "research label is contract compliance, not default within 90 days. Sensitive "
            "or proxy attributes require fairness and governance review before deployment.",
            "",
            "The champion is selected on validation PR-AUC. The test split remains sealed "
            "until model settings and the decision threshold are frozen.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_optimization(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    baseline_results_path: Path = DEFAULT_BASELINE_RESULTS,
    results_json: Path = DEFAULT_RESULTS_JSON,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    importance_csv: Path = DEFAULT_IMPORTANCE_CSV,
    explanations_json: Path = DEFAULT_EXPLANATIONS_JSON,
    report_path: Path = DEFAULT_REPORT,
    calibration_figure: Path = DEFAULT_CALIBRATION_FIGURE,
    importance_figure: Path = DEFAULT_IMPORTANCE_FIGURE,
    *,
    iterations: int = 8,
    cv_splits: int = 5,
    permutation_repeats: int = 20,
) -> OptimizationRun:
    """Run optimization without reading or evaluating the held-out test file."""

    training = _load_split(data_dir / "train.csv", "train")
    validation = _load_split(data_dir / "validation.csv", "validation")
    baseline = _read_baseline_results(baseline_results_path)
    run = optimize_and_evaluate(
        training,
        validation,
        iterations=iterations,
        cv_splits=cv_splits,
        permutation_repeats=permutation_repeats,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in run.models.items():
        joblib.dump(model, model_dir / f"optimized_{name}.joblib")
    joblib.dump(run.models[run.champion], model_dir / "optimized_champion.joblib")

    payload = {
        "experiment": "day_9_model_optimization",
        "dataset": "South German Credit research prototype",
        "target": TARGET_COLUMN,
        "cross_validation": run.cv_strategy,
        "search_scoring": "average_precision",
        "evaluation_split": "validation",
        "test_split_used": False,
        "decision_threshold": DECISION_THRESHOLD,
        "search_iterations_per_model": iterations,
        "champion": run.champion,
        "models": {
            name: {
                "best_cv_pr_auc": result.best_cv_pr_auc,
                "best_parameters": result.best_parameters,
                "validation_metrics": result.validation_metrics.to_dict(),
                "calibration": asdict(result.calibration),
            }
            for name, result in run.results.items()
        },
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    explanations_json.write_text(
        json.dumps(
            {
                "model": run.champion,
                "method": "single_feature_training_reference_perturbation",
                "records": run.explanations,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_results_csv(results_csv, run)
    _write_importance_csv(importance_csv, run)
    _write_markdown(report_path, run, baseline)
    _write_calibration_figure(
        calibration_figure,
        run.models,
        validation[FEATURE_COLUMNS],
        validation[TARGET_COLUMN].to_numpy(),
    )
    _write_importance_figure(importance_figure, run)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=20)
    args = parser.parse_args()
    run = run_optimization(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        iterations=args.iterations,
        cv_splits=args.cv_splits,
        permutation_repeats=args.permutation_repeats,
    )
    for name, result in run.results.items():
        metrics = result.validation_metrics
        print(
            f"{name}: CV PR-AUC={result.best_cv_pr_auc:.4f}, "
            f"validation ROC-AUC={metrics.roc_auc:.4f}, "
            f"validation PR-AUC={metrics.pr_auc:.4f}, "
            f"Brier={result.calibration.brier_score:.4f}"
        )
    print(f"Validation champion: {run.champion}")


if __name__ == "__main__":
    main()
