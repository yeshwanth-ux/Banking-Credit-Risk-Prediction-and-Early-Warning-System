"""Contract and behavior tests for the FastAPI prediction service."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from src.model_registry import RegisteredModelBundle
from src.risk_reporting import (
    PredictionResultWrite,
    RiskRecordFilters,
    ScoringModelMetadata,
    StoredScoringRun,
)


class FixedProbabilityPipeline:
    def __init__(self, probability: float = 0.60) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.tile([1 - self.probability, self.probability], (len(features), 1))


class FailingPipeline:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raise ValueError("internal model details")


class MemoryRiskRepository:
    def __init__(self) -> None:
        self.stored_results: list[PredictionResultWrite] = []
        self.stored_metadata: ScoringModelMetadata | None = None
        self.request_type: str | None = None
        self.last_filters: RiskRecordFilters | None = None

    def store_results(
        self,
        results,  # noqa: ANN001
        metadata: ScoringModelMetadata,
        *,
        request_type: str,
        requested_by: str,
    ) -> StoredScoringRun:
        self.stored_results = list(results)
        self.stored_metadata = metadata
        self.request_type = request_type
        assert requested_by == "banking-credit-risk-api"
        return StoredScoringRun(
            scoring_run_id=11,
            run_reference="run-test-11",
            prediction_ids=tuple(range(101, 101 + len(self.stored_results))),
            prediction_timestamp=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
        )

    def fetch_risk_records(
        self,
        filters: RiskRecordFilters,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.last_filters = filters
        assert limit == 25
        return [
            {
                "prediction_id": 101,
                "scoring_run_reference": "run-test-11",
                "reference_identifier": "LOAN-1001",
                "loan_reference": "LOAN-1001",
                "loan_status": "active",
                "prediction_timestamp": datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
                "model_name": "TestCreditRisk",
                "model_version": "3",
                "model_uri": "models:/TestCreditRisk@champion",
                "configuration_version": "test-config-1",
                "default_probability": Decimal("0.6000000"),
                "risk_band": "medium",
                "warning_reasons": [],
            }
        ]

    def portfolio_summary(self, filters: RiskRecordFilters) -> dict[str, Any]:
        self.last_filters = filters
        return {
            "total_accounts": 4,
            "risk_counts": {"high": 1, "medium": 2, "low": 1},
            "risk_percentages": {"high": 0.25, "medium": 0.5, "low": 0.25},
            "earliest_prediction_timestamp": datetime(
                2026, 10, 20, tzinfo=UTC
            ),
            "latest_prediction_timestamp": datetime(2026, 10, 27, tzinfo=UTC),
        }


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


def build_test_application(
    loader=model_bundle,  # noqa: ANN001
    *,
    repository: MemoryRiskRepository | None = None,
    api_key: str = "report-secret",
):  # noqa: ANN201
    selected_repository = repository or MemoryRiskRepository()
    return create_app(
        model_loader=loader,
        repository_factory=lambda: selected_repository,
        report_api_key=api_key,
    )


def prediction_payload(reference: str = "LOAN-1001") -> dict[str, Any]:
    return {
        "loan_reference": reference,
        "features": loan_features(),
        "warning_inputs": {},
    }


def test_health_reports_loaded_registry_version() -> None:
    application = build_test_application()

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
    repository = MemoryRiskRepository()
    application = build_test_application(repository=repository)
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
    assert body["scoring_run_reference"] == "run-test-11"
    assert body["prediction_id"] == 101
    assert repository.request_type == "single"
    assert repository.stored_results[0].reference_identifier == "LOAN-1001"
    assert repository.stored_metadata is not None
    assert repository.stored_metadata.model_version == "3"


def test_batch_prediction_preserves_order_and_count() -> None:
    repository = MemoryRiskRepository()
    application = build_test_application(
        loader=lambda: model_bundle(0.25),
        repository=repository,
    )
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
    assert repository.request_type == "batch"
    assert len(repository.stored_results) == 2


def test_request_validation_rejects_invalid_and_unknown_fields() -> None:
    application = build_test_application()
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

    application = build_test_application(loader=failed_loader)

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
    application = build_test_application(loader=lambda: bundle)

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/predict", json=prediction_payload())

    assert response.status_code == 500
    assert response.json() == {
        "error": "prediction_failed",
        "message": "The model could not score this loan",
    }
    assert "internal model details" not in response.text


def test_openapi_documents_prediction_endpoints_and_schemas() -> None:
    application = build_test_application()

    with TestClient(application) as client:
        response = client.get("/openapi.json")

    document = response.json()
    assert response.status_code == 200
    assert {"/health", "/predict", "/predict/batch"} <= set(document["paths"])
    assert "PredictionRequest" in document["components"]["schemas"]
    assert "PredictionResponse" in document["components"]["schemas"]
    assert "APIKeyHeader" in document["components"]["securitySchemes"]


def test_risk_records_require_authentication_and_exclude_sensitive_features() -> None:
    repository = MemoryRiskRepository()
    application = build_test_application(repository=repository)

    with TestClient(application) as client:
        missing = client.get("/risk-records")
        invalid = client.get("/risk-records", headers={"X-API-Key": "wrong"})
        permitted = client.get(
            "/risk-records",
            params={
                "prediction_from": "2026-10-01T00:00:00Z",
                "loan_status": "active",
                "risk_category": "medium",
                "limit": 25,
            },
            headers={"X-API-Key": "report-secret"},
        )

    assert missing.status_code == 403
    assert invalid.status_code == 403
    assert permitted.status_code == 200
    body = permitted.json()
    assert body["count"] == 1
    assert body["records"][0]["reference_identifier"] == "LOAN-1001"
    assert "feature_snapshot" not in body["records"][0]
    assert repository.last_filters is not None
    assert repository.last_filters.loan_status == "active"
    assert repository.last_filters.risk_category == "medium"


def test_portfolio_report_returns_validated_counts_and_filters() -> None:
    repository = MemoryRiskRepository()
    application = build_test_application(repository=repository)

    with TestClient(application) as client:
        response = client.get(
            "/reports/portfolio",
            params={"prediction_to": "2026-10-27T23:59:59Z"},
            headers={"X-API-Key": "report-secret"},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["total_accounts"] == 4
    assert body["risk_counts"] == {"high": 1, "medium": 2, "low": 1}
    assert sum(body["risk_counts"].values()) == body["total_accounts"]
    assert sum(body["risk_percentages"].values()) == 1.0
    assert body["filters"]["prediction_to"] == "2026-10-27T23:59:59Z"


def test_report_authentication_must_be_configured() -> None:
    application = build_test_application(api_key="")

    with TestClient(application) as client:
        response = client.get(
            "/reports/portfolio",
            headers={"X-API-Key": "anything"},
        )

    assert response.status_code == 503
    assert response.json()["error"] == "authentication_unavailable"
