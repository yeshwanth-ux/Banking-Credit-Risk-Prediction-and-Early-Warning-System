"""Run and verify the complete local credit-risk workflow."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from src.mlflow_tracking import sqlite_tracking_uri
from src.model_registry import load_registered_model
from src.pipeline import (
    DEFAULT_REPORT_ROOT,
    DEFAULT_RUNTIME_ROOT,
    DEFAULT_SOURCE,
    batch_predict_task,
    clean_data_task,
    generate_features_task,
    ingest_source_task,
    persist_predictions_task,
    pipeline_run_key,
    refresh_reports_task,
    run_directory,
    validate_source_task,
)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_end_to_end(
    *,
    source_path: Path = DEFAULT_SOURCE,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    logical_date: str,
    max_records: int = 10,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Execute the DAG task contract and verify API and report outputs."""

    validation = validate_source_task(source_path)
    run_key = pipeline_run_key(logical_date, validation["source_sha256"])
    runtime_directory = run_directory(run_key, runtime_root)
    report_directory = run_directory(run_key, report_root)

    ingestion = ingest_source_task(validation)
    cleaning = clean_data_task(
        ingestion,
        output_path=report_directory / "data_quality.md",
    )
    features = generate_features_task(
        cleaning,
        output_dir=runtime_directory / "features",
    )
    bundle = load_registered_model(
        tracking_uri=tracking_uri or sqlite_tracking_uri(),
    )
    prediction = batch_predict_task(
        features,
        output_path=runtime_directory / "predictions.json",
        max_records=max_records,
        bundle=bundle,
    )
    persistence = persist_predictions_task(
        prediction,
        idempotency_key=run_key,
    )
    reports = refresh_reports_task(
        persistence,
        output_directory=report_directory,
    )

    api_key = "local-e2e-analyst-key"
    application = create_app(
        model_loader=lambda: bundle,
        role_api_keys={api_key: {"risk_viewer", "risk_analyst", "monitor"}}
    )
    with TestClient(application) as client:
        health_response = client.get("/health")
        records_response = client.get(
            "/risk-records",
            params={"limit": min(max_records, 500)},
            headers={"X-API-Key": api_key},
        )
        portfolio_response = client.get(
            "/reports/portfolio",
            headers={"X-API-Key": api_key},
        )
        invalid_response = client.post(
            "/predict",
            json={"loan_reference": "INVALID-E2E", "features": {}},
        )

    if health_response.status_code != 200:
        raise RuntimeError("FastAPI health verification failed")
    if records_response.status_code != 200 or records_response.json()["count"] < 1:
        raise RuntimeError("FastAPI did not retrieve persisted risk records")
    if portfolio_response.status_code != 200:
        raise RuntimeError("FastAPI portfolio retrieval failed")
    if invalid_response.status_code != 422:
        raise RuntimeError("FastAPI invalid-input scenario did not return HTTP 422")

    summary_path = Path(reports["summary_csv"])
    high_risk_path = Path(reports["high_risk_csv"])
    trends_path = Path(reports["trends_csv"])
    html_path = Path(reports["html_report"])
    for artifact in (summary_path, high_risk_path, trends_path, html_path):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Expected report artifact is missing or empty: {artifact}")
    summary = pd.read_csv(summary_path)
    if int(summary["account_count"].sum()) != reports["latest_account_count"]:
        raise RuntimeError("Report summary does not reconcile to latest account count")

    manifest = {
        "completed_at_utc": datetime.now(tz=UTC).isoformat(),
        "logical_date": logical_date,
        "pipeline_run_key": run_key,
        "source": {
            "sha256": validation["source_sha256"],
            "rows_read": validation["rows_read"],
            "rows_accepted": validation["rows_accepted"],
            "rows_rejected": validation["rows_rejected"],
        },
        "database": {
            "ingestion_run_id": ingestion["run_id"],
            "inserted_records": ingestion["rows_inserted"],
            "updated_records": ingestion["rows_updated"],
            "rejected_records": ingestion["rows_rejected"],
            "scoring_run_id": persistence["scoring_run_id"],
            "persisted_predictions": persistence["record_count"],
        },
        "features": {
            "operational_rows": features["operational_rows"],
            "prototype_rows": features["prototype_rows"],
        },
        "model": {
            "version": prediction["model_version"],
            "scored_records": prediction["record_count"],
        },
        "api": {
            "health_status": health_response.status_code,
            "retrieved_records": records_response.json()["count"],
            "portfolio_total_accounts": portfolio_response.json()["total_accounts"],
            "invalid_input_status": invalid_response.status_code,
        },
        "reports": {
            "latest_account_count": reports["latest_account_count"],
            "high_risk_account_count": reports["high_risk_account_count"],
            "observed_period_count": reports["observed_period_count"],
            "trend_status": reports["trend_status"],
            "summary_csv": str(summary_path),
            "high_risk_csv": str(high_risk_path),
            "trends_csv": str(trends_path),
            "html_report": str(html_path),
        },
        "verification": "passed",
    }
    _write_manifest(report_directory / "end_to_end_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--logical-date",
        default=datetime.now(tz=UTC).date().isoformat(),
    )
    parser.add_argument("--max-records", type=int, default=10)
    parser.add_argument(
        "--tracking-uri",
        help="MLflow URI; defaults to the repository-local SQLite registry",
    )
    args = parser.parse_args()
    manifest = run_end_to_end(
        source_path=args.source,
        runtime_root=args.runtime_root,
        report_root=args.report_root,
        logical_date=args.logical_date,
        max_records=args.max_records,
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
