"""Track and register the frozen credit-risk model with local MLflow."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Some managed Windows hosts block optional SQLAlchemy C extensions. The pure
# Python implementation is fully supported for this small local MLflow store.
os.environ.setdefault("DISABLE_SQLALCHEMY_CEXT_RUNTIME", "1")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.data_preparation import (
    PROTOTYPE_CATEGORICAL_FEATURES,
    PROTOTYPE_NUMERIC_FEATURES,
)
from src.predict import DEFAULT_CONFIG, DEFAULT_FROZEN_MODEL, FEATURE_COLUMNS
from src.train import TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "mlflow.db"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"
DEFAULT_EXPERIMENT = "credit-risk-prototype"
DEFAULT_REGISTERED_MODEL = "CreditRiskEarlyWarning"
DEFAULT_MODEL_ALIAS = "champion"
DEFAULT_VALIDATION_DATA = PROJECT_ROOT / "data" / "processed" / "validation.csv"
DEFAULT_OPTIMIZATION_RESULTS = PROJECT_ROOT / "reports" / "optimized_model_results.json"
DEFAULT_SPLIT_SUMMARY = PROJECT_ROOT / "reports" / "dataset_split_summary.json"
DEFAULT_REGISTRY_REPORT = PROJECT_ROOT / "reports" / "mlflow_registered_model.json"

SUPPORTING_ARTIFACTS = (
    PROJECT_ROOT / "config" / "risk_classification.json",
    PROJECT_ROOT / "docs" / "data_dictionary.md",
    PROJECT_ROOT / "docs" / "data_preparation.md",
    PROJECT_ROOT / "docs" / "model_optimization.md",
    PROJECT_ROOT / "docs" / "early_warning.md",
    PROJECT_ROOT / "reports" / "optimized_model_results.json",
    PROJECT_ROOT / "reports" / "optimized_model_results.csv",
    PROJECT_ROOT / "reports" / "optimized_model_results.md",
    PROJECT_ROOT / "reports" / "permutation_importance.csv",
    PROJECT_ROOT / "reports" / "model_explanations.json",
    PROJECT_ROOT / "reports" / "risk_threshold_analysis.md",
    PROJECT_ROOT / "reports" / "figures" / "model_calibration.png",
    PROJECT_ROOT / "reports" / "figures" / "permutation_importance.png",
)


def sqlite_tracking_uri(database: Path = DEFAULT_DATABASE) -> str:
    """Return a portable absolute SQLAlchemy URI for a local SQLite store."""

    return f"sqlite:///{database.resolve().as_posix()}"


def local_tracking_uri(database: Path = DEFAULT_DATABASE) -> str:
    """Use an environment override or the project's local SQLite registry."""

    return os.getenv("MLFLOW_TRACKING_URI", sqlite_tracking_uri(database))


def configure_tracking(
    *,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> str:
    """Configure a local experiment and return its experiment ID."""

    uri = tracking_uri or local_tracking_uri()
    artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_root.resolve().as_uri(),
            tags={"project": "banking-credit-risk", "stage": "research-prototype"},
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def feature_definitions() -> dict[str, Any]:
    """Describe model inputs and their preprocessing contract."""

    return {
        "target": {
            "name": TARGET_COLUMN,
            "type": "binary integer",
            "meaning": "South German Credit bad-credit outcome (prototype proxy)",
        },
        "numerical_features": [
            {
                "name": name,
                "dtype": "numeric",
                "preprocessing": "median imputation then standard scaling",
            }
            for name in PROTOTYPE_NUMERIC_FEATURES
        ],
        "categorical_features": [
            {
                "name": name,
                "dtype": "categorical code",
                "preprocessing": "mode imputation then one-hot encoding",
            }
            for name in PROTOTYPE_CATEGORICAL_FEATURES
        ],
        "feature_count": len(FEATURE_COLUMNS),
        "leakage_policy": "Only model inputs available before the outcome are included.",
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_inputs(paths: tuple[Path, ...]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Day 7-10 artifacts: " + ", ".join(missing))


def _log_supporting_artifacts() -> None:
    for path in SUPPORTING_ARTIFACTS:
        if path.is_file():
            relative_parent = path.relative_to(PROJECT_ROOT).parent.as_posix()
            mlflow.log_artifact(str(path), artifact_path=f"evidence/{relative_parent}")


def track_selected_model(
    *,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL,
    alias: str = DEFAULT_MODEL_ALIAS,
    model_path: Path = DEFAULT_FROZEN_MODEL,
    config_path: Path = DEFAULT_CONFIG,
    validation_path: Path = DEFAULT_VALIDATION_DATA,
    optimization_results_path: Path = DEFAULT_OPTIMIZATION_RESULTS,
    split_summary_path: Path = DEFAULT_SPLIT_SUMMARY,
    registry_report_path: Path = DEFAULT_REGISTRY_REPORT,
) -> dict[str, Any]:
    """Log the frozen champion, register a version, and assign its service alias."""

    _validate_inputs(
        (model_path, config_path, validation_path, optimization_results_path, split_summary_path)
    )
    experiment_id = configure_tracking(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    config = _load_json(config_path)
    optimization = _load_json(optimization_results_path)
    split_summary = _load_json(split_summary_path)
    champion_name = optimization["champion"]
    champion = optimization["models"][champion_name]
    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "named_steps"):
        raise TypeError("Frozen model must be a fitted scikit-learn probability pipeline")
    if "preprocessor" not in pipeline.named_steps:
        raise ValueError("Frozen pipeline does not contain the required preprocessor step")

    validation = pd.read_csv(validation_path)
    missing_columns = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(validation.columns))
    if missing_columns:
        raise ValueError(f"Validation data is missing columns: {', '.join(missing_columns)}")
    features = validation[FEATURE_COLUMNS]
    probabilities = pipeline.predict_proba(features)[:, 1]
    prediction_frame = pd.DataFrame(
        {
            "source_record_id": validation["source_record_id"],
            "actual_bad_credit": validation[TARGET_COLUMN].astype(int),
            "predicted_probability": probabilities,
        }
    )
    training_configuration = {
        "experiment_source": optimization["experiment"],
        "dataset": optimization["dataset"],
        "target": optimization["target"],
        "champion": champion_name,
        "cross_validation": optimization["cross_validation"],
        "search_scoring": optimization["search_scoring"],
        "search_iterations_per_model": optimization["search_iterations_per_model"],
        "evaluation_split": optimization["evaluation_split"],
        "test_split_used": optimization["test_split_used"],
        "best_parameters": champion["best_parameters"],
        "split_definition": split_summary,
    }
    params: dict[str, Any] = {
        "model_family": config["model"]["family"],
        "model_version": config["model"]["version"],
        "configuration_version": config["configuration_version"],
        "training_rows": split_summary["counts"]["train"],
        "validation_rows": split_summary["counts"]["validation"],
        "raw_feature_count": len(FEATURE_COLUMNS),
        "preprocessor_fit_split": split_summary["preprocessor_fit_split"],
        "split_method": split_summary["method"],
        "search_scoring": optimization["search_scoring"],
        "medium_risk_threshold": config["risk_bands"]["medium_threshold"],
        "high_risk_threshold": config["risk_bands"]["high_threshold"],
    }
    params.update({f"best_{key}": value for key, value in champion["best_parameters"].items()})
    metrics = {
        f"validation_{key}": value
        for key, value in champion["validation_metrics"].items()
        if isinstance(value, (int, float))
    }
    metrics.update(
        {
            "validation_brier_score": champion["calibration"]["brier_score"],
            "validation_expected_calibration_error": champion["calibration"][
                "expected_calibration_error"
            ],
            "best_cv_pr_auc": champion["best_cv_pr_auc"],
        }
    )
    tags = {
        "model_stage": "research_prototype",
        "target_definition": config["target_definition"],
        "test_split_used": "false",
        "registered_for": "prediction_service",
        "frozen_model_sha256": config["model"]["sha256"],
    }

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{config['model']['version']}-registered",
        tags=tags,
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(feature_definitions(), "metadata/feature_definitions.json")
        mlflow.log_dict(training_configuration, "metadata/training_configuration.json")
        mlflow.log_dict(config, "metadata/risk_classification.json")
        mlflow.log_table(prediction_frame, "validation/predictions.json")
        _log_supporting_artifacts()

        signature = infer_signature(features, pipeline.predict(features))
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=features.head(3),
            metadata={
                "configuration_version": config["configuration_version"],
                "frozen_model_sha256": config["model"]["sha256"],
                "test_split_used": False,
            },
        )
        transformed = pipeline.named_steps["preprocessor"].transform(features.head(3))
        mlflow.sklearn.log_model(
            pipeline.named_steps["preprocessor"],
            name="preprocessor",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=infer_signature(features.head(3), transformed),
            input_example=features.head(3),
            metadata={"fitted_on": "training split only"},
        )
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    matching = [
        version
        for version in client.search_model_versions(f"name='{registered_model_name}'")
        if version.run_id == run_id
    ]
    if len(matching) != 1:
        raise RuntimeError("Could not resolve the newly registered model version")
    model_version = matching[0]
    version_tags = {
        "configuration_version": config["configuration_version"],
        "frozen_model_sha256": config["model"]["sha256"],
        "validation_pr_auc": str(champion["validation_metrics"]["pr_auc"]),
        "test_split_used": "false",
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(registered_model_name, model_version.version, key, value)
    client.update_model_version(
        name=registered_model_name,
        version=model_version.version,
        description=(
            "Frozen Day 10 logistic-regression research prototype. The registry artifact "
            "contains training-only preprocessing and the classifier. The held-out test "
            "split was not used for selection or threshold tuning."
        ),
    )
    client.set_registered_model_alias(registered_model_name, alias, model_version.version)

    active_tracking_uri = mlflow.get_tracking_uri()
    report_tracking_uri = (
        "sqlite:///mlflow.db"
        if active_tracking_uri == sqlite_tracking_uri(DEFAULT_DATABASE)
        else active_tracking_uri
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "tracking_uri": report_tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "registered_model_name": registered_model_name,
        "registered_model_version": model_version.version,
        "alias": alias,
        "model_uri": f"models:/{registered_model_name}@{alias}",
        "logged_model_uri": model_info.model_uri,
        "configuration_version": config["configuration_version"],
        "frozen_model_sha256": config["model"]["sha256"],
        "validation_pr_auc": champion["validation_metrics"]["pr_auc"],
        "test_split_used": False,
    }
    registry_report_path.parent.mkdir(parents=True, exist_ok=True)
    registry_report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--registered-model-name", default=DEFAULT_REGISTERED_MODEL)
    parser.add_argument("--alias", default=DEFAULT_MODEL_ALIAS)
    args = parser.parse_args()
    report = track_selected_model(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        alias=args.alias,
    )
    print(f"MLflow run: {report['run_id']}")
    print(f"Registered model: {report['model_uri']}")
    print(f"Tracking URI: {report['tracking_uri']}")


if __name__ == "__main__":
    main()
