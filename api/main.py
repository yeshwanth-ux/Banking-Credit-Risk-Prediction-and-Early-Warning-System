"""FastAPI prediction service for the registered credit-risk prototype."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.model_registry import RegisteredModelBundle, load_registered_model
from src.predict import RiskClassifier, RuleInputs, WarningResult

LOGGER = logging.getLogger(__name__)
SERVICE_NAME = "banking-credit-risk-api"


class ModelUnavailableError(RuntimeError):
    """Raised when prediction is requested before a model is available."""


class PredictionExecutionError(RuntimeError):
    """Raised when an unexpected model-inference failure occurs."""


class PredictionService:
    """Adapt the registered pipeline and warning classifier to API contracts."""

    def __init__(self, bundle: RegisteredModelBundle) -> None:
        self.bundle = bundle
        self.classifier: RiskClassifier = bundle.build_classifier()

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Assess one validated loan request."""

        try:
            result = self.classifier.assess(
                request.features.model_dump(),
                RuleInputs(**request.warning_inputs.model_dump()),
            )
        except (TypeError, ValueError) as error:
            raise PredictionExecutionError("The model could not score this loan") from error
        except Exception as error:
            LOGGER.exception("Unexpected model inference failure")
            raise PredictionExecutionError("Prediction processing failed") from error
        return _response_from_result(request.loan_reference, result)


def _response_from_result(
    loan_reference: str,
    result: WarningResult,
) -> PredictionResponse:
    return PredictionResponse(
        loan_reference=loan_reference,
        eligible=result.eligible,
        eligibility_reasons=list(result.eligibility_reasons),
        predicted_default_probability=result.risk_probability,
        probability_label=result.probability_label,
        model_risk_band=result.model_band,
        risk_band=result.final_band,
        model_reasons=list(result.model_reasons),
        model_explanations=[asdict(item) for item in result.model_explanations],
        warnings=[asdict(item) for item in result.rule_triggers],
        configuration_version=result.configuration_version,
        model_version=result.model_version,
        target_definition=result.target_definition,
    )


def _model_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise ModelUnavailableError("The registered model is not available")
    return service


def create_app(
    model_loader: Callable[[], RegisteredModelBundle] = load_registered_model,
) -> FastAPI:
    """Build the API, allowing an injected loader for isolated tests."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):  # noqa: ANN202
        application.state.prediction_service = None
        application.state.model_load_error = None
        try:
            bundle = model_loader()
            application.state.prediction_service = PredictionService(bundle)
        except Exception as error:
            application.state.model_load_error = type(error).__name__
            LOGGER.exception("Registered model failed to load during API startup")
        yield

    application = FastAPI(
        title="Banking Credit Risk Prediction API",
        summary="Prototype loan-risk predictions and early-warning indicators",
        description=(
            "Loads the MLflow `CreditRiskEarlyWarning@champion` pipeline and returns "
            "the prototype bad-credit probability, risk bands, explanations, and "
            "point-in-time warning rules. This research model is not a validated "
            "90-day probability-of-default model."
        ),
        version="1.0.0",
        lifespan=lifespan,
        contact={"name": "Credit Risk Model Team"},
        license_info={"name": "Research prototype"},
    )

    @application.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(
        request: Request,
        error: ModelUnavailableError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"error": "model_unavailable", "message": str(error)},
        )

    @application.exception_handler(PredictionExecutionError)
    async def prediction_error_handler(
        request: Request,
        error: PredictionExecutionError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": "prediction_failed", "message": str(error)},
        )

    unavailable_response: dict[int | str, dict[str, Any]] = {
        503: {"model": ErrorResponse, "description": "Registered model unavailable"},
        500: {"model": ErrorResponse, "description": "Prediction execution failed"},
    }

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["service"],
        summary="Check service and model readiness",
    )
    async def health(request: Request) -> HealthResponse | JSONResponse:
        service = getattr(request.app.state, "prediction_service", None)
        if service is None:
            payload = HealthResponse(
                status="unavailable",
                service=SERVICE_NAME,
                message="The registered model could not be loaded",
            )
            return JSONResponse(status_code=503, content=payload.model_dump())
        return HealthResponse(
            status="healthy",
            service=SERVICE_NAME,
            model_uri=service.bundle.model_uri,
            registered_model_version=service.bundle.registered_model_version,
            configuration_version=service.bundle.classification_config[
                "configuration_version"
            ],
        )

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        responses=unavailable_response,
        tags=["predictions"],
        summary="Predict risk for one loan",
    )
    async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        return _model_service(request).predict(payload)

    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses=unavailable_response,
        tags=["predictions"],
        summary="Predict risk for up to 100 loans",
    )
    async def predict_batch(
        payload: BatchPredictionRequest,
        request: Request,
    ) -> BatchPredictionResponse:
        service = _model_service(request)
        predictions = [service.predict(loan) for loan in payload.loans]
        return BatchPredictionResponse(count=len(predictions), predictions=predictions)

    return application


app = create_app()
