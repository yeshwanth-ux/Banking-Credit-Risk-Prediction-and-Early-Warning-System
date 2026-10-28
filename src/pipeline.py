"""Scheduler-independent task functions for the automated risk pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_processing import generate_quality_report
from src.feature_engineering import build_feature_datasets
from src.ingestion import ingest, read_source, validate_records
from src.model_registry import RegisteredModelBundle, load_registered_model
from src.predict import FEATURE_COLUMNS, RuleInputs
from src.risk_reporting import (
    PostgresRiskRepository,
    PredictionResultWrite,
    RiskRecordFilters,
    RiskRepository,
    ScoringModelMetadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "banking_data.csv"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "data" / "processed" / "airflow"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports" / "airflow"
LOGGER = logging.getLogger("credit_risk.pipeline")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def pipeline_run_key(logical_date: str, source_sha256: str) -> str:
    """Create a stable key so retries and repeated logical runs reuse results."""

    payload = f"credit-risk-daily:{logical_date}:{source_sha256}"
    return "credit-risk-daily:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_directory(run_key: str, runtime_root: Path = DEFAULT_RUNTIME_ROOT) -> Path:
    suffix = run_key.rsplit(":", maxsplit=1)[-1]
    return runtime_root / suffix


def validate_source_task(source_path: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    """Validate the raw file without mutating database state."""

    source = source_path.resolve()
    frame = read_source(source)
    validation = validate_records(frame)
    summary = {
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "rows_read": len(frame),
        "rows_accepted": len(validation.accepted),
        "rows_rejected": len(validation.rejected),
    }
    LOGGER.info("Source validation completed: %s", summary)
    if validation.rejected:
        raise ValueError(
            f"Source validation rejected {len(validation.rejected)} record(s)"
        )
    return summary


def ingest_source_task(validation_summary: dict[str, Any]) -> dict[str, Any]:
    """Upsert validated records into PostgreSQL."""

    summary = ingest(Path(validation_summary["source_path"]))
    result = asdict(summary)
    result["source_sha256"] = validation_summary["source_sha256"]
    LOGGER.info("Ingestion task completed: %s", result)
    return result


def clean_data_task(
    ingestion_summary: dict[str, Any],
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Refresh trusted views through their sources and enforce quality checks."""

    checks, metrics = generate_quality_report(output_path=output_path)
    failed = [check.name for check in checks if not check.passed]
    if failed:
        raise ValueError("Trusted dataset quality checks failed: " + ", ".join(failed))
    result = {
        "ingestion_run_id": ingestion_summary["run_id"],
        "quality_report": str(output_path.resolve()),
        "quality_checks": len(checks),
        "operational_rows": metrics.operational_rows,
        "prototype_rows": metrics.prototype_rows,
    }
    LOGGER.info("Cleaning and quality task completed: %s", result)
    return result


def generate_features_task(
    cleaning_summary: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate reusable operational and prototype feature datasets."""

    operational, prototype = build_feature_datasets(output_dir=output_dir)
    result = {
        "quality_report": cleaning_summary["quality_report"],
        "operational_features": str((output_dir / "operational_features.csv").resolve()),
        "prototype_features": str((output_dir / "prototype_features.csv").resolve()),
        "operational_rows": len(operational),
        "prototype_rows": len(prototype),
    }
    LOGGER.info("Feature generation completed: %s", result)
    return result


def batch_predict_task(
    feature_summary: dict[str, Any],
    *,
    output_path: Path,
    max_records: int | None = None,
    bundle: RegisteredModelBundle | None = None,
) -> dict[str, Any]:
    """Score a bounded feature file with the registered preprocessing pipeline."""

    selected_bundle = bundle or load_registered_model()
    classifier = selected_bundle.build_classifier()
    frame = pd.read_csv(feature_summary["prototype_features"])
    if max_records is not None:
        if max_records < 1:
            raise ValueError("max_records must be at least one")
        frame = frame.head(max_records)
    missing = sorted(set(["source_record_id", *FEATURE_COLUMNS]) - set(frame.columns))
    if missing:
        raise ValueError("Feature file is missing columns: " + ", ".join(missing))
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        feature_snapshot = {feature: row[feature] for feature in FEATURE_COLUMNS}
        result = classifier.assess(feature_snapshot, RuleInputs())
        if not result.eligible or result.risk_probability is None or result.final_band is None:
            raise ValueError(
                f"Feature record {row['source_record_id']} was not eligible for scoring"
            )
        records.append(
            {
                "reference_identifier": str(row["source_record_id"]),
                "probability": result.risk_probability,
                "risk_band": str(result.final_band),
                "warnings": [asdict(item) for item in result.rule_triggers],
                "feature_snapshot": feature_snapshot,
            }
        )
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "model": {
            "name": selected_bundle.registered_model_name,
            "version": selected_bundle.registered_model_version,
            "uri": selected_bundle.model_uri,
            "configuration_version": selected_bundle.classification_config[
                "configuration_version"
            ],
        },
        "record_count": len(records),
        "records": records,
    }
    _write_json(output_path, payload)
    summary = {
        "prediction_artifact": str(output_path.resolve()),
        "record_count": len(records),
        "model_version": selected_bundle.registered_model_version,
    }
    LOGGER.info("Batch prediction completed: %s", summary)
    return summary


def persist_predictions_task(
    prediction_summary: dict[str, Any],
    *,
    idempotency_key: str,
    repository: RiskRepository | None = None,
) -> dict[str, Any]:
    """Persist one idempotent batch scoring run."""

    payload = json.loads(
        Path(prediction_summary["prediction_artifact"]).read_text(encoding="utf-8")
    )
    records = [PredictionResultWrite(**record) for record in payload["records"]]
    model = payload["model"]
    metadata = ScoringModelMetadata(
        model_name=model["name"],
        model_version=model["version"],
        model_uri=model["uri"],
        configuration_version=model["configuration_version"],
    )
    selected_repository = repository or PostgresRiskRepository()
    stored = selected_repository.store_results(
        records,
        metadata,
        request_type="batch",
        requested_by="airflow-credit-risk-dag",
        idempotency_key=idempotency_key,
    )
    result = {
        "scoring_run_id": stored.scoring_run_id,
        "run_reference": stored.run_reference,
        "prediction_ids": list(stored.prediction_ids),
        "prediction_timestamp": stored.prediction_timestamp.isoformat(),
        "record_count": len(stored.prediction_ids),
        "idempotency_key": idempotency_key,
    }
    LOGGER.info("Prediction persistence completed: %s", result)
    return result


def refresh_reports_task(
    persistence_summary: dict[str, Any],
    *,
    output_path: Path,
    repository: RiskRepository | None = None,
) -> dict[str, Any]:
    """Regenerate the latest portfolio summary after successful persistence."""

    selected_repository = repository or PostgresRiskRepository()
    summary = selected_repository.portfolio_summary(RiskRecordFilters())
    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "source_scoring_run": persistence_summary["run_reference"],
        **summary,
    }
    _write_json(output_path, payload)
    result = {
        "portfolio_report": str(output_path.resolve()),
        "total_accounts": summary["total_accounts"],
        "risk_counts": summary["risk_counts"],
    }
    LOGGER.info("Portfolio report refreshed: %s", result)
    return result
