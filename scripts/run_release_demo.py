"""Generate anonymized ingestion, prediction, warning, and report demo outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.schemas import PredictionRequest
from src.ingestion import read_source, validate_records
from src.mlflow_tracking import sqlite_tracking_uri
from src.model_registry import load_registered_model
from src.predict import RuleInputs
from src.reporting import generate_portfolio_reports

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INGESTION_INPUT = PROJECT_ROOT / "demo" / "sample_ingestion.csv"
DEFAULT_PREDICTION_INPUT = PROJECT_ROOT / "demo" / "prediction_inputs.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "demo" / "output"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_demo(
    *,
    ingestion_input: Path = DEFAULT_INGESTION_INPUT,
    prediction_input: Path = DEFAULT_PREDICTION_INPUT,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    """Validate fresh sample rows, score synthetic loans, and export reports."""

    source = read_source(ingestion_input)
    validation = validate_records(source)
    if validation.rejected:
        raise ValueError("Release demo ingestion input contains rejected rows")

    requests = [
        PredictionRequest.model_validate(item)
        for item in json.loads(prediction_input.read_text(encoding="utf-8"))
    ]
    bundle = load_registered_model(tracking_uri=sqlite_tracking_uri())
    classifier = bundle.build_classifier()
    generated_at = datetime.now(tz=UTC)
    predictions: list[dict[str, Any]] = []
    report_records: list[dict[str, Any]] = []
    for index, request in enumerate(requests, start=1):
        result = classifier.assess(
            request.features.model_dump(),
            RuleInputs(**request.warning_inputs.model_dump()),
        )
        if not result.eligible or result.risk_probability is None or result.final_band is None:
            raise ValueError(f"Demo request was not eligible: {request.loan_reference}")
        warnings = [asdict(item) for item in result.rule_triggers]
        prediction = {
            "loan_reference": request.loan_reference,
            "predicted_bad_credit_probability": result.risk_probability,
            "probability_label": result.probability_label,
            "model_risk_band": str(result.model_band),
            "final_risk_band": str(result.final_band),
            "model_reasons": list(result.model_reasons),
            "rule_warnings": warnings,
            "model_version": bundle.registered_model_version,
            "configuration_version": result.configuration_version,
            "target_definition": result.target_definition,
        }
        predictions.append(prediction)
        report_records.append(
            {
                "prediction_id": index,
                "scoring_run_reference": "ANONYMIZED-DEMO-RUN",
                "reference_identifier": request.loan_reference,
                "loan_reference": request.loan_reference,
                "loan_status": "demonstration",
                "prediction_timestamp": generated_at,
                "model_name": bundle.registered_model_name,
                "model_version": bundle.registered_model_version,
                "model_uri": bundle.model_uri,
                "configuration_version": result.configuration_version,
                "default_probability": result.risk_probability,
                "risk_band": str(result.final_band),
                "warning_reasons": warnings,
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    reports = generate_portfolio_reports(
        output_directory,
        records=report_records,
        generated_at=generated_at,
    )
    report_paths = {
        "output_directory": reports.output_directory.relative_to(PROJECT_ROOT).as_posix(),
        "summary_csv": reports.summary_csv.relative_to(PROJECT_ROOT).as_posix(),
        "high_risk_csv": reports.high_risk_csv.relative_to(PROJECT_ROOT).as_posix(),
        "trends_csv": reports.trends_csv.relative_to(PROJECT_ROOT).as_posix(),
        "html_report": reports.html_report.relative_to(PROJECT_ROOT).as_posix(),
        "latest_account_count": reports.latest_account_count,
        "high_risk_account_count": reports.high_risk_account_count,
        "observed_period_count": reports.observed_period_count,
        "trend_status": reports.trend_status,
    }
    payload = {
        "demo_version": "1.0.0",
        "generated_at_utc": generated_at.isoformat(),
        "notice": (
            "Synthetic demonstration only; not real customers and not validated "
            "lending predictions."
        ),
        "ingestion": {
            "rows_read": len(source),
            "rows_accepted": len(validation.accepted),
            "rows_rejected": len(validation.rejected),
        },
        "predictions": predictions,
        "reports": report_paths,
    }
    _write_json(output_directory / "sample_predictions.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingestion-input", type=Path, default=DEFAULT_INGESTION_INPUT)
    parser.add_argument("--prediction-input", type=Path, default=DEFAULT_PREDICTION_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    result = run_demo(
        ingestion_input=args.ingestion_input,
        prediction_input=args.prediction_input,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
