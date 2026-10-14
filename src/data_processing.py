"""Create and validate trusted point-in-time analytical datasets."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from psycopg import Connection

from src.database import DatabaseSettings, connection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "data_quality_weekly.md"


@dataclass(frozen=True)
class QualityCheck:
    name: str
    failed_rows: int
    description: str

    @property
    def passed(self) -> bool:
        return self.failed_rows == 0


@dataclass(frozen=True)
class DatasetMetrics:
    operational_rows: int
    operational_loans: int
    prototype_rows: int
    prototype_bad_credit_rate: Decimal | None


def clean_identifier(value: Any) -> str:
    """Normalize a source identifier without changing its semantic characters."""

    cleaned = re.sub(r"\s+", "-", str(value).strip().upper())
    cleaned = re.sub(r"[^A-Z0-9._-]", "", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        raise ValueError("Identifier is empty after cleaning")
    return cleaned


def clean_optional_text(value: Any) -> str | None:
    """Trim and collapse whitespace; map blank values to null."""

    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value).strip())
    return cleaned or None


def seed_application_prediction_dates(db: Connection) -> int:
    """Create one idempotent underwriting snapshot per loan application."""

    result = db.execute(
        """
        INSERT INTO loan_prediction_dates (loan_id, prediction_date, snapshot_source)
        SELECT loan_id, application_date, 'application'
        FROM loans
        ON CONFLICT (loan_id, prediction_date) DO NOTHING
        """
    )
    return result.rowcount


def add_prediction_date(
    db: Connection,
    loan_id: int,
    prediction_date: date,
    snapshot_source: str = "scheduled",
) -> None:
    """Upsert an explicitly requested loan snapshot date."""

    db.execute(
        """
        INSERT INTO loan_prediction_dates (loan_id, prediction_date, snapshot_source)
        VALUES (%s, %s, %s)
        ON CONFLICT (loan_id, prediction_date) DO UPDATE
        SET snapshot_source = EXCLUDED.snapshot_source
        """,
        (loan_id, prediction_date, snapshot_source),
    )


QUALITY_QUERIES = (
    (
        "operational_unique_key",
        """
        SELECT COUNT(*) FROM (
            SELECT loan_id, prediction_date
            FROM trusted_loan_analytical_dataset
            GROUP BY loan_id, prediction_date HAVING COUNT(*) > 1
        ) duplicates
        """,
        "Each operational record has a unique loan and prediction date.",
    ),
    (
        "operational_required_fields",
        """
        SELECT COUNT(*) FROM trusted_loan_analytical_dataset
        WHERE loan_id IS NULL OR prediction_date IS NULL OR customer_id IS NULL
           OR principal_amount IS NULL OR term_months IS NULL
        """,
        "Operational identifiers and required loan values are populated.",
    ),
    (
        "operational_valid_ranges",
        """
        SELECT COUNT(*) FROM trusted_loan_analytical_dataset
        WHERE principal_amount <= 0 OR term_months <= 0
           OR interest_rate_pct NOT BETWEEN 0 AND 100
           OR installments_due < 0 OR installments_paid < 0
           OR installments_overdue < 0 OR maximum_days_past_due < 0
           OR paid_amount_as_of_date < 0 OR outstanding_amount_as_of_date < 0
        """,
        "Operational numeric features remain within valid ranges.",
    ),
    (
        "operational_valid_keys",
        """
        SELECT COUNT(*) FROM loan_prediction_dates p
        LEFT JOIN loans l ON l.loan_id = p.loan_id
        LEFT JOIN customers c ON c.customer_id = l.customer_id
        WHERE l.loan_id IS NULL OR c.customer_id IS NULL
        """,
        "Every prediction date resolves to a valid loan and customer.",
    ),
    (
        "operational_prediction_timing",
        """
        SELECT COUNT(*) FROM loan_prediction_dates p
        JOIN loans l ON l.loan_id = p.loan_id
        WHERE p.prediction_date < l.application_date
        """,
        "No prediction precedes its loan application.",
    ),
    (
        "prototype_unique_key",
        """
        SELECT COUNT(*) FROM (
            SELECT source_record_id FROM trusted_credit_risk_dataset
            GROUP BY source_record_id HAVING COUNT(*) > 1
        ) duplicates
        """,
        "Each prototype record has a unique deterministic source key.",
    ),
    (
        "prototype_required_fields",
        """
        SELECT COUNT(*) FROM trusted_credit_risk_dataset
        WHERE source_record_id IS NULL OR duration_months IS NULL
           OR credit_amount_dm IS NULL OR age_years IS NULL OR bad_credit IS NULL
        """,
        "Required prototype model fields are populated.",
    ),
    (
        "prototype_valid_ranges",
        """
        SELECT COUNT(*) FROM trusted_credit_risk_dataset
        WHERE duration_months <= 0 OR credit_amount_dm <= 0
           OR age_years NOT BETWEEN 18 AND 120 OR bad_credit NOT IN (0, 1)
        """,
        "Prototype numeric values and label remain within valid ranges.",
    ),
)


def run_quality_checks(db: Connection) -> list[QualityCheck]:
    checks: list[QualityCheck] = []
    for name, query, description in QUALITY_QUERIES:
        row = db.execute(query).fetchone()
        if row is None:
            raise RuntimeError(f"Quality query returned no result: {name}")
        checks.append(QualityCheck(name, int(row["count"]), description))
    return checks


def collect_metrics(db: Connection) -> DatasetMetrics:
    row = db.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM trusted_loan_analytical_dataset) AS operational_rows,
            (SELECT COUNT(DISTINCT loan_id) FROM trusted_loan_analytical_dataset)
                AS operational_loans,
            (SELECT COUNT(*) FROM trusted_credit_risk_dataset) AS prototype_rows,
            (SELECT AVG(bad_credit) FROM trusted_credit_risk_dataset)
                AS prototype_bad_credit_rate
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("Dataset metrics query returned no result")
    return DatasetMetrics(
        operational_rows=int(row["operational_rows"]),
        operational_loans=int(row["operational_loans"]),
        prototype_rows=int(row["prototype_rows"]),
        prototype_bad_credit_rate=row["prototype_bad_credit_rate"],
    )


def render_quality_report(
    checks: list[QualityCheck],
    metrics: DatasetMetrics,
    report_date: date,
) -> str:
    passed = sum(check.passed for check in checks)
    rate = (
        "n/a"
        if metrics.prototype_bad_credit_rate is None
        else f"{metrics.prototype_bad_credit_rate:.2%}"
    )
    lines = [
        "# Weekly dataset quality summary",
        "",
        f"**Report date:** {report_date.isoformat()}",
        f"**Overall status:** {'PASS' if passed == len(checks) else 'FAIL'}",
        f"**Checks passed:** {passed}/{len(checks)}",
        "",
        "## Dataset coverage",
        "",
        "| Dataset | Rows | Loans | Bad-credit rate |",
        "|---|---:|---:|---:|",
        f"| Operational point-in-time dataset | {metrics.operational_rows:,} "
        f"| {metrics.operational_loans:,} | n/a |",
        f"| South German Credit prototype | {metrics.prototype_rows:,} | n/a | {rate} |",
        "",
        "## Quality checks",
        "",
        "| Check | Status | Failed rows | Rule |",
        "|---|---|---:|---|",
    ]
    for check in checks:
        lines.append(
            f"| `{check.name}` | {'PASS' if check.passed else 'FAIL'} "
            f"| {check.failed_rows:,} | {check.description} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The operational dataset contains one row per loan and prediction date. "
            "Repayment aggregates include only installments due on or before that date, "
            "and payments made after it are excluded from features.",
            "",
            "The South German Credit prototype has no event dates or repayment history. "
            "Its `bad_credit` outcome supports exploratory classification only and must "
            "not be interpreted as the project's 90-day default target.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_quality_report(
    output_path: Path = DEFAULT_REPORT,
    settings: DatabaseSettings | None = None,
    *,
    seed_application_dates: bool = False,
) -> tuple[list[QualityCheck], DatasetMetrics]:
    with connection(settings) as db:
        if seed_application_dates:
            seed_application_prediction_dates(db)
        checks = run_quality_checks(db)
        metrics = collect_metrics(db)
        report_date_row = db.execute("SELECT CURRENT_DATE AS report_date").fetchone()
        if report_date_row is None:
            raise RuntimeError("Database did not return a report date")
        report_date = report_date_row["report_date"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_quality_report(checks, metrics, report_date),
        encoding="utf-8",
    )
    return checks, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed-application-dates", action="store_true")
    args = parser.parse_args()
    checks, metrics = generate_quality_report(
        args.output,
        seed_application_dates=args.seed_application_dates,
    )
    failed = [check for check in checks if not check.passed]
    print(
        f"Quality checks: {len(checks) - len(failed)}/{len(checks)} passed; "
        f"operational rows: {metrics.operational_rows:,}; "
        f"prototype rows: {metrics.prototype_rows:,}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
