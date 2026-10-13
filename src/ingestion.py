"""Validate and ingest raw credit-risk CSV records into PostgreSQL."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
from psycopg import Connection
from psycopg.types.json import Jsonb

from src.database import DatabaseSettings, connection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "banking_data.csv"
LOGGER = logging.getLogger("credit_risk.ingestion")

COLUMNS = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount_dm", "savings", "employment_duration", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence", "property",
    "age_years", "other_installment_plans", "housing", "number_credits", "job",
    "people_liable", "telephone", "foreign_worker", "credit_risk", "bad_credit",
]

CATEGORY_RANGES: dict[str, set[int]] = {
    "checking_account_status": {1, 2, 3, 4},
    "credit_history": {0, 1, 2, 3, 4},
    "purpose": set(range(11)),
    "savings": {1, 2, 3, 4, 5},
    "employment_duration": {1, 2, 3, 4, 5},
    "installment_rate": {1, 2, 3, 4},
    "personal_status_sex": {1, 2, 3, 4},
    "other_debtors": {1, 2, 3},
    "present_residence": {1, 2, 3, 4},
    "property": {1, 2, 3, 4},
    "other_installment_plans": {1, 2, 3},
    "housing": {1, 2, 3},
    "number_credits": {1, 2, 3, 4},
    "job": {1, 2, 3, 4},
    "people_liable": {1, 2},
    "telephone": {1, 2},
    "foreign_worker": {1, 2},
    "credit_risk": {0, 1},
    "bad_credit": {0, 1},
}


@dataclass(frozen=True)
class RejectedRow:
    row_number: int
    record: dict[str, Any]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ValidationResult:
    accepted: list[dict[str, Any]]
    rejected: list[RejectedRow]


@dataclass(frozen=True)
class IngestionSummary:
    run_id: int
    rows_read: int
    rows_accepted: int
    rows_rejected: int
    rows_inserted: int
    rows_updated: int


def standardize_column_name(value: str) -> str:
    """Return a predictable snake_case column identifier."""

    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return re.sub(r"_+", "_", value)


def read_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Raw source file does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV source files are supported")
    data = pd.read_csv(path, dtype="string", keep_default_na=True)
    data.columns = [standardize_column_name(str(column)) for column in data.columns]
    if data.columns.duplicated().any():
        raise ValueError("Standardized source contains duplicate column names")
    return data


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _integer(value: Any) -> int:
    number = Decimal(str(value).strip())
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError
    return int(number)


def _record_id(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_financial_fields(record: dict[str, Any]) -> list[str]:
    """Validate optional common financial and repayment fields when supplied."""

    errors: list[str] = []
    for field in ("principal_amount", "scheduled_amount", "paid_amount"):
        if record.get(field) not in (None, ""):
            try:
                amount = Decimal(str(record[field]))
                if not amount.is_finite() or amount < 0 or (
                    field == "principal_amount" and amount == 0
                ):
                    errors.append(f"invalid_{field}")
            except InvalidOperation:
                errors.append(f"invalid_{field}")
    if record.get("interest_rate_pct") not in (None, ""):
        try:
            rate = Decimal(str(record["interest_rate_pct"]))
            if not rate.is_finite() or not Decimal("0") <= rate <= Decimal("100"):
                errors.append("invalid_interest_rate_pct")
        except InvalidOperation:
            errors.append("invalid_interest_rate_pct")

    parsed_dates: dict[str, date] = {}
    date_fields = (
        "application_date",
        "origination_date",
        "maturity_date",
        "due_date",
        "payment_date",
    )
    for field in date_fields:
        if record.get(field) not in (None, ""):
            try:
                parsed_dates[field] = date.fromisoformat(str(record[field]))
            except ValueError:
                errors.append(f"invalid_{field}")
    originated = parsed_dates.get("origination_date")
    if originated and parsed_dates.get("application_date", originated) > originated:
        errors.append("origination_before_application")
    if originated and parsed_dates.get("maturity_date", originated) < originated:
        errors.append("maturity_before_origination")
    if originated and parsed_dates.get("due_date", originated) < originated:
        errors.append("due_date_before_origination")
    if originated and parsed_dates.get("payment_date", originated) < originated:
        errors.append("payment_date_before_origination")
    return errors


def validate_records(data: pd.DataFrame) -> ValidationResult:
    missing_columns = sorted(set(COLUMNS) - set(data.columns))
    unexpected_columns = sorted(set(data.columns) - set(COLUMNS))
    if missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append(f"missing: {', '.join(missing_columns)}")
        if unexpected_columns:
            details.append(f"unexpected: {', '.join(unexpected_columns)}")
        raise ValueError(
            "Source columns do not match the data dictionary (" + "; ".join(details) + ")"
        )

    accepted: list[dict[str, Any]] = []
    rejected: list[RejectedRow] = []
    seen: set[str] = set()
    for offset, (_, row) in enumerate(data.iterrows(), start=2):
        raw = {column: _json_value(row[column]) for column in COLUMNS}
        errors = [f"missing_{column}" for column, value in raw.items() if value is None]
        normalized: dict[str, Any] = {}
        if not errors:
            for column, value in raw.items():
                try:
                    normalized[column] = _integer(value)
                except (InvalidOperation, ValueError):
                    errors.append(f"invalid_integer_{column}")
            for column, allowed in CATEGORY_RANGES.items():
                if column in normalized and normalized[column] not in allowed:
                    errors.append(f"invalid_category_{column}")
            if normalized.get("duration_months", 0) <= 0:
                errors.append("invalid_duration_months")
            if normalized.get("credit_amount_dm", 0) <= 0:
                errors.append("invalid_credit_amount_dm")
            if not 18 <= normalized.get("age_years", 0) <= 120:
                errors.append("invalid_age_years")
            if normalized.get("bad_credit") != 1 - normalized.get("credit_risk", -1):
                errors.append("target_mismatch")

        identity_source = normalized if normalized else raw
        source_record_id = _record_id(identity_source)
        if source_record_id in seen:
            errors.append("duplicate_record")
        seen.add(source_record_id)

        if errors:
            rejected.append(RejectedRow(offset, raw, tuple(sorted(set(errors)))))
        else:
            normalized["source_record_id"] = source_record_id
            normalized["source_row_number"] = offset
            accepted.append(normalized)
    return ValidationResult(accepted, rejected)


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_run(db: Connection, path: Path) -> int:
    row = db.execute(
        "INSERT INTO ingestion_runs (source_name, source_sha256) VALUES (%s, %s) "
        "RETURNING ingestion_run_id",
        (path.name, _source_sha256(path)),
    ).fetchone()
    if row is None:
        raise RuntimeError("PostgreSQL did not return an ingestion run ID")
    return int(row["ingestion_run_id"])


def ingest(
    path: Path = DEFAULT_SOURCE,
    settings: DatabaseSettings | None = None,
) -> IngestionSummary:
    data = read_source(path)
    result = validate_records(data)

    with connection(settings) as db:
        run_id = _create_run(db, path)

    inserted = updated = 0
    try:
        with connection(settings) as db:
            for rejected in result.rejected:
                db.execute(
                    "INSERT INTO ingestion_rejections "
                    "(ingestion_run_id, source_row_number, source_record, error_codes) "
                    "VALUES (%s, %s, %s, %s)",
                    (run_id, rejected.row_number, Jsonb(rejected.record), Jsonb(rejected.errors)),
                )

            value_columns = ", ".join(COLUMNS)
            placeholders = ", ".join(["%s"] * (len(COLUMNS) + 3))
            assignments = ", ".join(f"{name} = EXCLUDED.{name}" for name in COLUMNS)
            statement = (
                "INSERT INTO credit_risk_observations "
                f"(source_record_id, source_row_number, {value_columns}, ingestion_run_id) "
                f"VALUES ({placeholders}) ON CONFLICT (source_record_id) DO UPDATE SET "
                f"source_row_number = EXCLUDED.source_row_number, {assignments}, "
                "ingestion_run_id = EXCLUDED.ingestion_run_id, ingested_at = CURRENT_TIMESTAMP "
                "RETURNING (xmax = 0) AS inserted"
            )
            for record in result.accepted:
                parameters = [record["source_record_id"], record["source_row_number"]]
                parameters.extend(record[column] for column in COLUMNS)
                parameters.append(run_id)
                outcome = db.execute(statement, parameters).fetchone()
                if outcome and outcome["inserted"]:
                    inserted += 1
                else:
                    updated += 1

            db.execute(
                "UPDATE ingestion_runs SET completed_at = CURRENT_TIMESTAMP, status = 'completed', "
                "rows_read = %s, rows_accepted = %s, rows_rejected = %s, "
                "rows_inserted = %s, rows_updated = %s WHERE ingestion_run_id = %s",
                (len(data), len(result.accepted), len(result.rejected), inserted, updated, run_id),
            )
    except Exception as error:
        LOGGER.exception("Ingestion run %s failed", run_id)
        with connection(settings) as db:
            db.execute(
                "UPDATE ingestion_runs SET completed_at = CURRENT_TIMESTAMP, status = 'failed', "
                "rows_read = %s, rows_accepted = %s, rows_rejected = %s, error_message = %s "
                "WHERE ingestion_run_id = %s",
                (len(data), len(result.accepted), len(result.rejected), str(error), run_id),
            )
        raise

    summary = IngestionSummary(
        run_id, len(data), len(result.accepted), len(result.rejected), inserted, updated
    )
    LOGGER.info("Ingestion completed: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    summary = ingest(args.source)
    print(json.dumps(summary.__dict__, indent=2))


if __name__ == "__main__":
    main()
