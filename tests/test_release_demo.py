"""Release-demo input contracts."""

from __future__ import annotations

import json
from pathlib import Path

from api.schemas import PredictionRequest
from src.ingestion import read_source, validate_records

ROOT = Path(__file__).resolve().parents[1]


def test_release_demo_ingestion_rows_are_valid_and_anonymized() -> None:
    result = validate_records(read_source(ROOT / "demo" / "sample_ingestion.csv"))

    assert len(result.accepted) == 3
    assert result.rejected == []


def test_release_demo_prediction_requests_are_schema_valid_and_anonymized() -> None:
    payload = json.loads((ROOT / "demo" / "prediction_inputs.json").read_text())
    requests = [PredictionRequest.model_validate(item) for item in payload]

    assert len(requests) == 3
    assert all(request.loan_reference.startswith("DEMO-") for request in requests)
    assert len({request.loan_reference for request in requests}) == 3


def test_release_demo_outputs_cover_all_risk_bands_without_machine_paths() -> None:
    output_path = ROOT / "demo" / "output" / "sample_predictions.json"
    source = output_path.read_text(encoding="utf-8")
    output = json.loads(source)

    assert "Synthetic demonstration only" in output["notice"]
    assert output["ingestion"] == {
        "rows_read": 3,
        "rows_accepted": 3,
        "rows_rejected": 0,
    }
    assert {item["final_risk_band"] for item in output["predictions"]} == {
        "low",
        "medium",
        "high",
    }
    assert output["reports"]["high_risk_account_count"] == 1
    assert output["reports"]["trend_status"] == "insufficient_observed_periods"
    assert "C:\\Users" not in source
