"""FastAPI prediction service for the registered credit-risk prototype."""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PortfolioSummaryResponse,
    PredictionRequest,
    PredictionResponse,
    ReportFiltersResponse,
    RiskRecordListResponse,
    RiskRecordResponse,
)
from src.model_registry import RegisteredModelBundle, load_registered_model
from src.predict import RiskClassifier, RuleInputs, WarningResult
from src.risk_reporting import (
    LOAN_STATUSES,
    PostgresRiskRepository,
    PredictionResultWrite,
    RiskRecordFilters,
    RiskRepository,
    ScoringModelMetadata,
)

LOGGER = logging.getLogger(__name__)
SERVICE_NAME = "banking-credit-risk-api"


class ModelUnavailableError(RuntimeError):
    """Raised when prediction is requested before a model is available."""


class PredictionExecutionError(RuntimeError):
    """Raised when an unexpected model-inference failure occurs."""


class PersistenceError(RuntimeError):
    """Raised when prediction storage or reporting fails."""


class ReportAuthenticationError(RuntimeError):
    """Raised when protected reporting credentials are missing or invalid."""


class ReportAuthenticationUnavailableError(RuntimeError):
    """Raised when report authentication has not been configured."""


class PredictionService:
    """Adapt the registered pipeline and warning classifier to API contracts."""

    def __init__(
        self,
        bundle: RegisteredModelBundle,
        repository: RiskRepository,
    ) -> None:
        self.bundle = bundle
        self.repository = repository
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

    def persist(
        self,
        requests: list[PredictionRequest],
        responses: list[PredictionResponse],
        *,
        request_type: str,
    ) -> list[PredictionResponse]:
        """Store one atomic scoring run and attach persistence metadata."""

        writes = [
            PredictionResultWrite(
                reference_identifier=request.loan_reference,
                probability=float(response.predicted_default_probability),
                risk_band=str(response.risk_band),
                warnings=[warning.model_dump(mode="json") for warning in response.warnings],
                feature_snapshot=request.features.model_dump(mode="json"),
            )
            for request, response in zip(requests, responses, strict=True)
            if response.predicted_default_probability is not None
            and response.risk_band is not None
        ]
        if len(writes) != len(responses):
            raise PersistenceError("Ineligible prediction results cannot be stored")
        metadata = ScoringModelMetadata(
            model_name=self.bundle.registered_model_name,
            model_version=self.bundle.registered_model_version,
            model_uri=self.bundle.model_uri,
            configuration_version=self.bundle.classification_config[
                "configuration_version"
            ],
        )
        try:
            stored = self.repository.store_results(
                writes,
                metadata,
                request_type=request_type,
                requested_by=SERVICE_NAME,
            )
        except Exception as error:
            LOGGER.exception("Failed to store scoring results")
            raise PersistenceError("Prediction results could not be stored") from error
        return [
            response.model_copy(
                update={
                    "scoring_run_reference": stored.run_reference,
                    "prediction_id": prediction_id,
                    "prediction_timestamp": stored.prediction_timestamp,
                }
            )
            for response, prediction_id in zip(
                responses,
                stored.prediction_ids,
                strict=True,
            )
        ]


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
    repository_factory: Callable[[], RiskRepository] = PostgresRiskRepository,
    report_api_key: str | None = None,
) -> FastAPI:
    """Build the API, allowing an injected loader for isolated tests."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):  # noqa: ANN202
        application.state.prediction_service = None
        application.state.model_load_error = None
        try:
            bundle = model_loader()
            application.state.prediction_service = PredictionService(
                bundle,
                repository_factory(),
            )
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
    configured_report_api_key = (
        report_api_key
        if report_api_key is not None
        else os.getenv("RISK_REPORT_API_KEY")
    )
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authorize_reports(
        provided_key: str | None = Security(api_key_header),
    ) -> str:
        if not configured_report_api_key:
            raise ReportAuthenticationUnavailableError(
                "Report authentication is not configured"
            )
        if provided_key is None or not secrets.compare_digest(
            provided_key,
            configured_report_api_key,
        ):
            raise ReportAuthenticationError("A valid report API key is required")
        return "api-key"

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

    @application.exception_handler(PersistenceError)
    async def persistence_error_handler(
        request: Request,
        error: PersistenceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": "persistence_failed", "message": str(error)},
        )

    @application.exception_handler(ReportAuthenticationError)
    async def report_authentication_handler(
        request: Request,
        error: ReportAuthenticationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={"error": "forbidden", "message": str(error)},
        )

    @application.exception_handler(ReportAuthenticationUnavailableError)
    async def report_authentication_unavailable_handler(
        request: Request,
        error: ReportAuthenticationUnavailableError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"error": "authentication_unavailable", "message": str(error)},
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
        service = _model_service(request)
        response = service.predict(payload)
        return service.persist([payload], [response], request_type="single")[0]

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
        stored = service.persist(
            payload.loans,
            predictions,
            request_type="batch",
        )
        return BatchPredictionResponse(count=len(stored), predictions=stored)

    def report_filters(
        prediction_from: datetime | None,
        prediction_to: datetime | None,
        loan_status: str | None,
        risk_category: str | None,
    ) -> RiskRecordFilters:
        filters = RiskRecordFilters(
            prediction_from=prediction_from,
            prediction_to=prediction_to,
            loan_status=loan_status,
            risk_category=risk_category,
        )
        try:
            filters.validate()
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return filters

    @application.get(
        "/risk-records",
        response_model=RiskRecordListResponse,
        responses={403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["reports"],
        summary="Retrieve permitted latest risk records",
    )
    async def risk_records(
        request: Request,
        principal: str = Security(authorize_reports),
        prediction_from: datetime | None = None,
        prediction_to: datetime | None = None,
        loan_status: str | None = Query(default=None, enum=LOAN_STATUSES),
        risk_category: str | None = Query(default=None, enum=("low", "medium", "high")),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> RiskRecordListResponse:
        service = _model_service(request)
        filters = report_filters(
            prediction_from,
            prediction_to,
            loan_status,
            risk_category,
        )
        try:
            rows = service.repository.fetch_risk_records(filters, limit=limit)
        except Exception as error:
            LOGGER.exception("Failed to retrieve risk records")
            raise PersistenceError("Risk records could not be retrieved") from error
        records = [
            RiskRecordResponse(
                prediction_id=row["prediction_id"],
                scoring_run_reference=row["scoring_run_reference"],
                reference_identifier=row["reference_identifier"],
                loan_reference=row["loan_reference"],
                loan_status=row["loan_status"],
                prediction_timestamp=row["prediction_timestamp"],
                model_name=row["model_name"],
                model_version=row["model_version"],
                model_uri=row["model_uri"],
                configuration_version=row["configuration_version"],
                predicted_default_probability=float(row["default_probability"]),
                risk_band=row["risk_band"],
                warnings=row["warning_reasons"],
            )
            for row in rows
        ]
        return RiskRecordListResponse(count=len(records), records=records)

    @application.get(
        "/reports/portfolio",
        response_model=PortfolioSummaryResponse,
        responses={403: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["reports"],
        summary="Summarize latest portfolio risk distribution",
    )
    async def portfolio_report(
        request: Request,
        principal: str = Security(authorize_reports),
        prediction_from: datetime | None = None,
        prediction_to: datetime | None = None,
        loan_status: str | None = Query(default=None, enum=LOAN_STATUSES),
        risk_category: str | None = Query(default=None, enum=("low", "medium", "high")),
    ) -> PortfolioSummaryResponse:
        service = _model_service(request)
        filters = report_filters(
            prediction_from,
            prediction_to,
            loan_status,
            risk_category,
        )
        try:
            summary = service.repository.portfolio_summary(filters)
        except Exception as error:
            LOGGER.exception("Failed to create portfolio report")
            raise PersistenceError("Portfolio report could not be created") from error
        return PortfolioSummaryResponse(
            **summary,
            filters=ReportFiltersResponse(
                prediction_from=filters.prediction_from,
                prediction_to=filters.prediction_to,
                loan_status=filters.loan_status,
                risk_category=filters.risk_category,
            ),
        )

    return application


app = create_app()
