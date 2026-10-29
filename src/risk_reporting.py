"""Persist scoring results and query restricted portfolio risk summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol
from uuid import uuid4

from psycopg import Connection
from psycopg.types.json import Jsonb

from src.database import DatabaseSettings, connection

LOAN_STATUSES = (
    "application",
    "approved",
    "active",
    "delinquent",
    "defaulted",
    "paid_off",
    "charged_off",
    "cancelled",
)
RISK_BANDS = ("low", "medium", "high")


@dataclass(frozen=True)
class PredictionResultWrite:
    reference_identifier: str
    probability: float
    risk_band: str
    warnings: list[dict[str, Any]]
    feature_snapshot: dict[str, Any]


@dataclass(frozen=True)
class ScoringModelMetadata:
    model_name: str
    model_version: str
    model_uri: str
    configuration_version: str


@dataclass(frozen=True)
class StoredScoringRun:
    scoring_run_id: int
    run_reference: str
    prediction_ids: tuple[int, ...]
    prediction_timestamp: datetime


@dataclass(frozen=True)
class RiskRecordFilters:
    prediction_from: datetime | None = None
    prediction_to: datetime | None = None
    loan_status: str | None = None
    risk_category: str | None = None

    def validate(self) -> None:
        if (
            self.prediction_from is not None
            and self.prediction_to is not None
            and self.prediction_from > self.prediction_to
        ):
            raise ValueError("prediction_from must be before or equal to prediction_to")
        if self.loan_status is not None and self.loan_status not in LOAN_STATUSES:
            raise ValueError("Unsupported loan status")
        if self.risk_category is not None and self.risk_category not in RISK_BANDS:
            raise ValueError("Unsupported risk category")


class RiskRepository(Protocol):
    def store_results(
        self,
        results: Sequence[PredictionResultWrite],
        metadata: ScoringModelMetadata,
        *,
        request_type: str,
        requested_by: str,
        idempotency_key: str | None = None,
    ) -> StoredScoringRun: ...

    def fetch_risk_records(
        self,
        filters: RiskRecordFilters,
        *,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    def portfolio_summary(self, filters: RiskRecordFilters) -> dict[str, Any]: ...

    def fetch_scoring_history(
        self,
        filters: RiskRecordFilters,
    ) -> list[dict[str, Any]]: ...


def _filter_sql(filters: RiskRecordFilters) -> tuple[list[str], list[Any]]:
    filters.validate()
    clauses: list[str] = []
    parameters: list[Any] = []
    if filters.prediction_from is not None:
        clauses.append("prediction.prediction_timestamp >= %s")
        parameters.append(filters.prediction_from)
    if filters.prediction_to is not None:
        clauses.append("prediction.prediction_timestamp <= %s")
        parameters.append(filters.prediction_to)
    if filters.loan_status is not None:
        clauses.append("loan.loan_status = %s")
        parameters.append(filters.loan_status)
    return clauses, parameters


def _latest_predictions_cte(filters: RiskRecordFilters) -> tuple[str, list[Any]]:
    clauses, parameters = _filter_sql(filters)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return (
        f"""
        WITH ranked_predictions AS (
            SELECT
                prediction.prediction_id,
                prediction.scoring_run_id,
                scoring_run.run_reference AS scoring_run_reference,
                prediction.reference_identifier,
                loan.loan_reference,
                loan.loan_status,
                prediction.prediction_timestamp,
                prediction.model_name,
                prediction.model_version,
                prediction.model_uri,
                prediction.configuration_version,
                prediction.default_probability,
                prediction.risk_band,
                prediction.warning_reasons,
                ROW_NUMBER() OVER (
                    PARTITION BY prediction.reference_identifier
                    ORDER BY
                        prediction.prediction_timestamp DESC,
                        prediction.prediction_id DESC
                ) AS result_rank
            FROM risk_predictions AS prediction
            LEFT JOIN loans AS loan ON loan.loan_id = prediction.loan_id
            LEFT JOIN scoring_runs AS scoring_run
                ON scoring_run.scoring_run_id = prediction.scoring_run_id
            {where_sql}
        ),
        latest_predictions AS (
            SELECT * FROM ranked_predictions WHERE result_rank = 1
        )
        """,
        parameters,
    )


def fetch_risk_records_query(
    filters: RiskRecordFilters,
    *,
    limit: int,
) -> tuple[str, list[Any]]:
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    cte, parameters = _latest_predictions_cte(filters)
    risk_clause = ""
    if filters.risk_category is not None:
        risk_clause = "WHERE risk_band = %s"
        parameters.append(filters.risk_category)
    parameters.append(limit)
    return (
        cte
        + f"""
        SELECT
            prediction_id,
            scoring_run_reference,
            reference_identifier,
            loan_reference,
            loan_status,
            prediction_timestamp,
            model_name,
            model_version,
            model_uri,
            configuration_version,
            default_probability,
            risk_band,
            warning_reasons
        FROM latest_predictions
        {risk_clause}
        ORDER BY prediction_timestamp DESC, prediction_id DESC
        LIMIT %s
        """,
        parameters,
    )


def portfolio_summary_query(filters: RiskRecordFilters) -> tuple[str, list[Any]]:
    cte, parameters = _latest_predictions_cte(filters)
    risk_clause = ""
    if filters.risk_category is not None:
        risk_clause = "WHERE risk_band = %s"
        parameters.append(filters.risk_category)
    return (
        cte
        + f"""
        SELECT
            COUNT(*)::INTEGER AS total_accounts,
            COUNT(*) FILTER (WHERE risk_band = 'high')::INTEGER AS high_risk_count,
            COUNT(*) FILTER (WHERE risk_band = 'medium')::INTEGER AS medium_risk_count,
            COUNT(*) FILTER (WHERE risk_band = 'low')::INTEGER AS low_risk_count,
            MIN(prediction_timestamp) AS earliest_prediction_timestamp,
            MAX(prediction_timestamp) AS latest_prediction_timestamp
        FROM latest_predictions
        {risk_clause}
        """,
        parameters,
    )


def scoring_history_query(filters: RiskRecordFilters) -> tuple[str, list[Any]]:
    """Select report-safe scoring history without feature snapshots or customer data."""

    clauses, parameters = _filter_sql(filters)
    if filters.risk_category is not None:
        clauses.append("prediction.risk_band = %s")
        parameters.append(filters.risk_category)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return (
        f"""
        SELECT
            prediction.prediction_id,
            scoring_run.run_reference AS scoring_run_reference,
            prediction.reference_identifier,
            loan.loan_reference,
            loan.loan_status,
            prediction.prediction_timestamp,
            prediction.model_name,
            prediction.model_version,
            prediction.model_uri,
            prediction.configuration_version,
            prediction.default_probability,
            prediction.risk_band,
            prediction.warning_reasons
        FROM risk_predictions AS prediction
        LEFT JOIN loans AS loan ON loan.loan_id = prediction.loan_id
        LEFT JOIN scoring_runs AS scoring_run
            ON scoring_run.scoring_run_id = prediction.scoring_run_id
        {where_sql}
        ORDER BY prediction.prediction_timestamp, prediction.prediction_id
        """,
        parameters,
    )


def execute_risk_records_query(
    db_connection: Connection,
    filters: RiskRecordFilters,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    statement, parameters = fetch_risk_records_query(filters, limit=limit)
    return list(db_connection.execute(statement, parameters).fetchall())


def execute_portfolio_summary_query(
    db_connection: Connection,
    filters: RiskRecordFilters,
) -> dict[str, Any]:
    statement, parameters = portfolio_summary_query(filters)
    row = db_connection.execute(statement, parameters).fetchone()
    if row is None:
        raise RuntimeError("Portfolio aggregation did not return a result")
    total = int(row["total_accounts"])
    counts = {
        "high": int(row["high_risk_count"]),
        "medium": int(row["medium_risk_count"]),
        "low": int(row["low_risk_count"]),
    }
    percentages = {
        band: (count / total if total else 0.0) for band, count in counts.items()
    }
    return {
        "total_accounts": total,
        "risk_counts": counts,
        "risk_percentages": percentages,
        "earliest_prediction_timestamp": row["earliest_prediction_timestamp"],
        "latest_prediction_timestamp": row["latest_prediction_timestamp"],
    }


def execute_scoring_history_query(
    db_connection: Connection,
    filters: RiskRecordFilters,
) -> list[dict[str, Any]]:
    statement, parameters = scoring_history_query(filters)
    return list(db_connection.execute(statement, parameters).fetchall())


class PostgresRiskRepository:
    """PostgreSQL persistence with aggregate queries that omit sensitive features."""

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self.settings = settings

    def store_results(
        self,
        results: Sequence[PredictionResultWrite],
        metadata: ScoringModelMetadata,
        *,
        request_type: str,
        requested_by: str,
        idempotency_key: str | None = None,
    ) -> StoredScoringRun:
        if request_type not in {"single", "batch"}:
            raise ValueError("request_type must be single or batch")
        if not results:
            raise ValueError("At least one prediction result is required")
        if idempotency_key is not None and not idempotency_key.strip():
            raise ValueError("idempotency_key cannot be blank")
        references = [result.reference_identifier for result in results]
        if len(set(references)) != len(references):
            raise ValueError("Loan references must be unique within a scoring run")
        for result in results:
            if result.risk_band not in RISK_BANDS:
                raise ValueError("Prediction result has an invalid risk band")
            if not 0 <= result.probability <= 1:
                raise ValueError("Prediction probability must be between zero and one")

        timestamp = datetime.now(tz=UTC)
        run_reference = str(uuid4())
        with connection(self.settings) as db_connection:
            loans = db_connection.execute(
                "SELECT loan_id, loan_reference FROM loans WHERE loan_reference = ANY(%s)",
                (references,),
            ).fetchall()
            loan_ids = {row["loan_reference"]: row["loan_id"] for row in loans}
            scoring_run = db_connection.execute(
                """
                INSERT INTO scoring_runs (
                    run_reference, request_type, started_at, completed_at, status,
                    model_name, model_version, model_uri, configuration_version,
                    requested_by, requested_count, succeeded_count, failed_count,
                    idempotency_key
                )
                VALUES (
                    %s, %s, %s, %s, 'completed', %s, %s, %s, %s,
                    %s, %s, %s, 0, %s
                )
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING scoring_run_id
                """,
                (
                    run_reference,
                    request_type,
                    timestamp,
                    timestamp,
                    metadata.model_name,
                    metadata.model_version,
                    metadata.model_uri,
                    metadata.configuration_version,
                    requested_by,
                    len(results),
                    len(results),
                    idempotency_key,
                ),
            ).fetchone()
            if scoring_run is None:
                if idempotency_key is None:
                    raise RuntimeError("Failed to create scoring run")
                existing = db_connection.execute(
                    """
                    SELECT
                        scoring_run_id, run_reference, completed_at,
                        request_type, model_name, model_version, model_uri,
                        configuration_version, requested_count
                    FROM scoring_runs
                    WHERE idempotency_key = %s
                    """,
                    (idempotency_key,),
                ).fetchone()
                if existing is None:
                    raise RuntimeError("Idempotent scoring run could not be resolved")
                expected = (
                    request_type,
                    metadata.model_name,
                    metadata.model_version,
                    metadata.model_uri,
                    metadata.configuration_version,
                    len(results),
                )
                actual = (
                    existing["request_type"],
                    existing["model_name"],
                    existing["model_version"],
                    existing["model_uri"],
                    existing["configuration_version"],
                    existing["requested_count"],
                )
                if actual != expected:
                    raise ValueError("Idempotency key was already used for another scoring run")
                prediction_rows = db_connection.execute(
                    """
                    SELECT prediction_id, prediction_timestamp
                    FROM risk_predictions
                    WHERE scoring_run_id = %s
                    ORDER BY prediction_id
                    """,
                    (existing["scoring_run_id"],),
                ).fetchall()
                if len(prediction_rows) != len(results):
                    raise RuntimeError("Existing scoring run has an incomplete result set")
                return StoredScoringRun(
                    scoring_run_id=existing["scoring_run_id"],
                    run_reference=existing["run_reference"],
                    prediction_ids=tuple(row["prediction_id"] for row in prediction_rows),
                    prediction_timestamp=prediction_rows[0]["prediction_timestamp"],
                )
            prediction_ids: list[int] = []
            for result in results:
                prediction = db_connection.execute(
                    """
                    INSERT INTO risk_predictions (
                        loan_id, scoring_run_id, reference_identifier,
                        prediction_timestamp, model_name, model_version, model_uri,
                        configuration_version, default_probability, risk_band,
                        warning_reasons, feature_snapshot
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING prediction_id
                    """,
                    (
                        loan_ids.get(result.reference_identifier),
                        scoring_run["scoring_run_id"],
                        result.reference_identifier,
                        timestamp,
                        metadata.model_name,
                        metadata.model_version,
                        metadata.model_uri,
                        metadata.configuration_version,
                        Decimal(str(result.probability)),
                        result.risk_band,
                        Jsonb(result.warnings),
                        Jsonb(result.feature_snapshot),
                    ),
                ).fetchone()
                if prediction is None:
                    raise RuntimeError("Failed to store prediction result")
                prediction_ids.append(prediction["prediction_id"])
        return StoredScoringRun(
            scoring_run_id=scoring_run["scoring_run_id"],
            run_reference=run_reference,
            prediction_ids=tuple(prediction_ids),
            prediction_timestamp=timestamp,
        )

    def fetch_risk_records(
        self,
        filters: RiskRecordFilters,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        with connection(self.settings) as db_connection:
            return execute_risk_records_query(db_connection, filters, limit=limit)

    def portfolio_summary(self, filters: RiskRecordFilters) -> dict[str, Any]:
        with connection(self.settings) as db_connection:
            return execute_portfolio_summary_query(db_connection, filters)

    def fetch_scoring_history(
        self,
        filters: RiskRecordFilters,
    ) -> list[dict[str, Any]]:
        with connection(self.settings) as db_connection:
            return execute_scoring_history_query(db_connection, filters)
