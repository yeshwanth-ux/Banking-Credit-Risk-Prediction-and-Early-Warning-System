"""Pydantic request and response contracts for the credit-risk API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApiModel(BaseModel):
    """Base API contract with strict field names and useful OpenAPI examples."""

    model_config = ConfigDict(extra="forbid", strict=True)


class LoanFeatures(ApiModel):
    """Features required by the registered South German Credit prototype."""

    duration_months: int = Field(gt=0, le=120)
    credit_amount_dm: float = Field(gt=0)
    age_years: int = Field(ge=18, le=120)
    credit_amount_per_month: float = Field(gt=0)
    checking_account_status: int = Field(ge=1, le=4)
    credit_history: int = Field(ge=0, le=4)
    purpose: int = Field(ge=0, le=10)
    savings: int = Field(ge=1, le=5)
    employment_duration: int = Field(ge=1, le=5)
    installment_rate: int = Field(ge=1, le=4)
    personal_status_sex: int = Field(ge=1, le=4)
    other_debtors: int = Field(ge=1, le=3)
    present_residence: int = Field(ge=1, le=4)
    property: int = Field(ge=1, le=4)
    other_installment_plans: int = Field(ge=1, le=3)
    housing: int = Field(ge=1, le=3)
    number_credits: int = Field(ge=1, le=4)
    job: int = Field(ge=1, le=4)
    people_liable: int = Field(ge=1, le=2)
    telephone: int = Field(ge=1, le=2)
    foreign_worker: int = Field(ge=1, le=2)
    young_borrower_flag: Literal[0, 1]
    long_duration_flag: Literal[0, 1]

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "example": {
                "duration_months": 24,
                "credit_amount_dm": 4000.0,
                "age_years": 35,
                "credit_amount_per_month": 166.67,
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
        },
    )


class WarningInputs(ApiModel):
    """Point-in-time history used by deterministic early-warning rules."""

    missed_payment_count: int = Field(default=0, ge=0)
    previous_missed_payment_count: int = Field(default=0, ge=0)
    maximum_days_past_due: int = Field(default=0, ge=0)
    previous_maximum_days_past_due: int = Field(default=0, ge=0)
    debt_to_income_ratio: float | None = Field(default=None, ge=0)
    previous_debt_to_income_ratio: float | None = Field(default=None, ge=0)
    previous_default_count: int = Field(default=0, ge=0)


class PredictionRequest(ApiModel):
    """One loan assessment request."""

    loan_reference: str = Field(min_length=1, max_length=100)
    features: LoanFeatures
    warning_inputs: WarningInputs = Field(default_factory=WarningInputs)


class BatchPredictionRequest(ApiModel):
    """Bounded collection of loan assessments."""

    loans: list[PredictionRequest] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def references_are_unique(self) -> BatchPredictionRequest:
        references = [loan.loan_reference for loan in self.loans]
        if len(references) != len(set(references)):
            raise ValueError("loan_reference values must be unique within a batch")
        return self


class ModelExplanationResponse(ApiModel):
    feature: str
    direction: Literal["increased", "decreased"]
    probability_effect: float
    reason: str


class WarningIndicatorResponse(ApiModel):
    code: str
    severity: Literal["low", "medium", "high"]
    reason: str
    evidence: dict[str, Any]


class PredictionResponse(ApiModel):
    loan_reference: str
    eligible: bool
    eligibility_reasons: list[str]
    predicted_default_probability: float | None = Field(default=None, ge=0, le=1)
    probability_label: str
    model_risk_band: Literal["low", "medium", "high"] | None
    risk_band: Literal["low", "medium", "high"] | None
    model_reasons: list[str]
    model_explanations: list[ModelExplanationResponse]
    warnings: list[WarningIndicatorResponse]
    configuration_version: str
    model_version: str
    target_definition: str
    scoring_run_reference: str | None = None
    prediction_id: int | None = None
    prediction_timestamp: datetime | None = None


class BatchPredictionResponse(ApiModel):
    count: int = Field(ge=0)
    predictions: list[PredictionResponse]


class HealthResponse(ApiModel):
    status: Literal["healthy", "unavailable"]
    service: str
    model_uri: str | None = None
    registered_model_version: str | None = None
    configuration_version: str | None = None
    message: str | None = None


class ErrorResponse(ApiModel):
    error: str
    message: str


class RiskRecordResponse(ApiModel):
    prediction_id: int
    scoring_run_reference: str | None
    reference_identifier: str
    loan_reference: str | None
    loan_status: str | None
    prediction_timestamp: datetime
    model_name: str
    model_version: str
    model_uri: str
    configuration_version: str
    predicted_default_probability: float = Field(ge=0, le=1)
    risk_band: Literal["low", "medium", "high"]
    warnings: list[dict[str, Any]]


class RiskRecordListResponse(ApiModel):
    count: int = Field(ge=0)
    records: list[RiskRecordResponse]


class RiskCounts(ApiModel):
    high: int = Field(ge=0)
    medium: int = Field(ge=0)
    low: int = Field(ge=0)


class RiskPercentages(ApiModel):
    high: float = Field(ge=0, le=1)
    medium: float = Field(ge=0, le=1)
    low: float = Field(ge=0, le=1)


class ReportFiltersResponse(ApiModel):
    prediction_from: datetime | None
    prediction_to: datetime | None
    loan_status: str | None
    risk_category: Literal["low", "medium", "high"] | None


class PortfolioSummaryResponse(ApiModel):
    total_accounts: int = Field(ge=0)
    risk_counts: RiskCounts
    risk_percentages: RiskPercentages
    earliest_prediction_timestamp: datetime | None
    latest_prediction_timestamp: datetime | None
    filters: ReportFiltersResponse
