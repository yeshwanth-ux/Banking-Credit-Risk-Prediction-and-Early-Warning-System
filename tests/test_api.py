"""Contract and behavior tests for the FastAPI prediction service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from src.model_registry import RegisteredModelBundle


class FixedProbabilityPipeline:
    def __init__(self, probability: float = 0.60) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile([1 - self.probability, self.probability], (len(features), 1))


class FailingPipeline:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raise ValueError("internal model details")


def loan_features() -> dict[str, Any]:
    return {
        "duration_months": 24,
        "credit_amount_dm": 4000.0,
        "age_years": 35,
        "credit_amount_per_month": 4000 / 24,
        "checking_account_status": 2,
        "credit_history": 2,
        "purpose": 3,
        "savings": 2,
        "employment_duration": 3,
        "installment_rate": 3,
        "personal_status_sex": 3,
        "other_debtors": 1,
        "present_residence": 3,
        "property": 2,
        "other_installment_plans": 3,
        "housing": 2,
        "number_credits": 1,
        "job": 3,
        "people_liable": 2,
        "telephone": 2,
        "foreign_worker": 1,
        "young_borrower_flag": 0,
        "long_duration_flag": 0,
    }


def classification_config() -> dict[str, Any]:
    return {
        "configuration_version": "test-config-1",
        "model": {"version": "test-model-1"},
        "target_definition": "prototype bad-credit target",
        "probability_label": "prototype bad-credit risk probability",
        "risk_bands": {"medium_threshold": 0.40, "high_threshold": 0.80},
        "warning_rules": {
            "severe_delinquency_days": 90,
            "worsening_delay_days": 15,
            "high_debt_to_income_ratio": 0.40,
            "debt_to_income_increase": 0.10,
        },
        "explanation_reference_values": loan_features(),
    }


def model_bundle(probability: float = 0.60) -> RegisteredModelBundle:
    return RegisteredModelBundle(
        pipeline=FixedProbabilityPipeline(probability),
        classification_config=classification_config(),
        model_uri="models:/TestCreditRisk@champion",
        registered_model_name="TestCreditRisk",
        registered_model_version="3",
        alias="champion",
        run_id="test-run",
    )


def prediction_payload(reference: str = "LOAN-1001") -> dict[str, Any]:
    return {
        "loan_reference": reference,
        "features": loan_features(),
        "warning_inputs": {},
    }


def test_health_reports_loaded_registry_version() -> None:
    application = create_app(model_loader=model_bundle)

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "banking-credit-risk-api",
        "model_uri": "models:/TestCreditRisk@champion",
        "registered_model_version": "3",
        "configuration_version": "test-config-1",
        "message": None,
    }


def test_single_prediction_returns_probability_band_and_warning() -> None:
    application = create_app(model_loader=model_bundle)
    payload = prediction_payload()
    payload["warning_inputs"] = {"maximum_days_past_due": 95}

    with TestClient(application) as client:
        response = client.post("/predict", json=payload)

    body = response.json()
    assert response.status_code == 200
    assert body["loan_reference"] == "LOAN-1001"
    assert body["predicted_default_probability"] == 0.60
    assert body["model_risk_band"] == "medium"
    assert body["risk_band"] == "high"
    assert [warning["code"] for warning in body["warnings"]] == ["severe_delinquency"]
    assert body["model_reasons"]
    assert body["target_definition"] == "prototype bad-credit target"


def test_batch_prediction_preserves_order_and_count() -> None:
    application = create_app(model_loader=lambda: model_bundle(0.25))
    payload = {"loans": [prediction_payload("LOAN-A"), prediction_payload("LOAN-B")]}

    with TestClient(application) as client:
        response = client.post("/predict/batch", json=payload)

    body = response.json()
    assert response.status_code == 200
    assert body["count"] == 2
    assert [item["loan_reference"] for item in body["predictions"]] == [
        "LOAN-A",
        "LOAN-B",
    ]
    assert all(item["risk_band"] == "low" for item in body["predictions"])


def test_request_validation_rejects_invalid_and_unknown_fields() -> None:
    application = create_app(model_loader=model_bundle)
    payload = prediction_payload()
    payload["features"]["age_years"] = 15
    payload["features"]["unknown_feature"] = 1

    with TestClient(application) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    locations = {tuple(error["loc"]) for error in response.json()["detail"]}
    assert ("body", "features", "age_years") in locations
    assert ("body", "features", "unknown_feature") in locations


def test_model_load_failure_keeps_health_available_and_blocks_predictions() -> None:
    def failed_loader() -> RegisteredModelBundle:
        raise RuntimeError("registry offline")

    application = create_app(model_loader=failed_loader)

    with TestClient(application) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=prediction_payload())

    assert health.status_code == 503
    assert health.json()["status"] == "unavailable"
    assert "registry offline" not in health.text
    assert prediction.status_code == 503
    assert prediction.json() == {
        "error": "model_unavailable",
        "message": "The registered model is not available",
    }


def test_prediction_failure_returns_structured_error_without_internal_details() -> None:
    bundle = model_bundle()
    bundle = RegisteredModelBundle(
        pipeline=FailingPipeline(),
        classification_config=bundle.classification_config,
        model_uri=bundle.model_uri,
        registered_model_name=bundle.registered_model_name,
        registered_model_version=bundle.registered_model_version,
        alias=bundle.alias,
        run_id=bundle.run_id,
    )
    application = create_app(model_loader=lambda: bundle)

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/predict", json=prediction_payload())

    assert response.status_code == 500
    assert response.json() == {
        "error": "prediction_failed",
        "message": "The model could not score this loan",
    }
    assert "internal model details" not in response.text


def test_openapi_documents_prediction_endpoints_and_schemas() -> None:
    application = create_app(model_loader=model_bundle)

    with TestClient(application) as client:
        response = client.get("/openapi.json")

    document = response.json()
    assert response.status_code == 200
    assert {"/health", "/predict", "/predict/batch"} <= set(document["paths"])
    assert "PredictionRequest" in document["components"]["schemas"]
    assert "PredictionResponse" in document["components"]["schemas"]
