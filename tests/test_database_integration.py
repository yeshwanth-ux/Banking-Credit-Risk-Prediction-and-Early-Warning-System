from __future__ import annotations

import os
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from psycopg.types.json import Jsonb

from src.database import connection
from src.feature_engineering import OPERATIONAL_SOURCE_QUERY

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="Set RUN_POSTGRES_TESTS=1 to run PostgreSQL integration tests",
)


def test_core_insert_and_query_flow() -> None:
    unique_suffix = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S%f")

    with connection() as db_connection:
        with db_connection.transaction(force_rollback=True):
            customer = db_connection.execute(
                """
                INSERT INTO customers (customer_reference, annual_income, country_code)
                VALUES (%s, %s, %s)
                RETURNING customer_id
                """,
                (f"TEST-CUSTOMER-{unique_suffix}", Decimal("72000.00"), "US"),
            ).fetchone()
            assert customer is not None

            loan = db_connection.execute(
                """
                INSERT INTO loans (
                    loan_reference, customer_id, application_date, origination_date,
                    maturity_date, principal_amount, interest_rate_pct, term_months,
                    purpose, loan_status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING loan_id
                """,
                (
                    f"TEST-LOAN-{unique_suffix}",
                    customer["customer_id"],
                    date(2026, 1, 1),
                    date(2026, 1, 2),
                    date(2027, 1, 2),
                    Decimal("10000.00"),
                    Decimal("7.2500"),
                    12,
                    "test",
                    "active",
                ),
            ).fetchone()
            assert loan is not None

            db_connection.execute(
                """
                INSERT INTO repayment_history (
                    loan_id, installment_number, due_date, payment_date,
                    scheduled_amount, paid_amount, repayment_status
                )
                VALUES (%s, 1, %s, %s, %s, %s, 'paid')
                """,
                (
                    loan["loan_id"],
                    date(2026, 2, 2),
                    date(2026, 2, 2),
                    Decimal("850.00"),
                    Decimal("850.00"),
                ),
            )
            db_connection.execute(
                """
                INSERT INTO account_activity (
                    customer_id, loan_id, source_event_reference,
                    activity_timestamp, activity_type, amount
                )
                VALUES (%s, %s, %s, %s, 'payment', %s)
                """,
                (
                    customer["customer_id"],
                    loan["loan_id"],
                    f"TEST-EVENT-{unique_suffix}",
                    datetime(2026, 2, 2, tzinfo=UTC),
                    Decimal("850.00"),
                ),
            )
            db_connection.execute(
                """
                INSERT INTO risk_predictions (
                    loan_id, prediction_timestamp, model_name, model_version,
                    default_probability, risk_band, warning_reasons, feature_snapshot
                )
                VALUES (%s, %s, 'integration-test', '1', %s, 'low', %s, %s)
                """,
                (
                    loan["loan_id"],
                    datetime(2026, 2, 3, tzinfo=UTC),
                    Decimal("0.1250000"),
                    Jsonb([]),
                    Jsonb({"test_feature": 1}),
                ),
            )

            result = db_connection.execute(
                """
                SELECT
                    c.customer_reference,
                    l.loan_reference,
                    COUNT(DISTINCT r.repayment_id) AS repayments,
                    COUNT(DISTINCT a.activity_id) AS activities,
                    COUNT(DISTINCT p.prediction_id) AS predictions
                FROM customers c
                JOIN loans l ON l.customer_id = c.customer_id
                LEFT JOIN repayment_history r ON r.loan_id = l.loan_id
                LEFT JOIN account_activity a ON a.loan_id = l.loan_id
                LEFT JOIN risk_predictions p ON p.loan_id = l.loan_id
                WHERE l.loan_id = %s
                GROUP BY c.customer_reference, l.loan_reference
                """,
                (loan["loan_id"],),
            ).fetchone()

            assert result is not None
            assert result["repayments"] == 1
            assert result["activities"] == 1
            assert result["predictions"] == 1


def test_trusted_dataset_excludes_future_repayments() -> None:
    unique_suffix = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S%f")

    with connection() as db_connection:
        with db_connection.transaction(force_rollback=True):
            customer = db_connection.execute(
                """
                INSERT INTO customers (customer_reference, birth_date, annual_income)
                VALUES (%s, %s, %s) RETURNING customer_id
                """,
                (f"DQ-CUSTOMER-{unique_suffix}", date(1990, 1, 1), Decimal("60000")),
            ).fetchone()
            assert customer is not None
            loan = db_connection.execute(
                """
                INSERT INTO loans (
                    loan_reference, customer_id, application_date, origination_date,
                    principal_amount, interest_rate_pct, term_months, loan_status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
                RETURNING loan_id
                """,
                (
                    f"DQ-LOAN-{unique_suffix}",
                    customer["customer_id"],
                    date(2026, 1, 1),
                    date(2026, 1, 2),
                    Decimal("1200"),
                    Decimal("8"),
                    12,
                ),
            ).fetchone()
            assert loan is not None
            db_connection.execute(
                """
                INSERT INTO repayment_history (
                    loan_id, installment_number, due_date, payment_date,
                    scheduled_amount, paid_amount, repayment_status
                ) VALUES
                    (%s, 1, %s, %s, 100, 100, 'late'),
                    (%s, 2, %s, %s, 100, 100, 'paid')
                """,
                (
                    loan["loan_id"],
                    date(2026, 2, 1),
                    date(2026, 3, 20),
                    loan["loan_id"],
                    date(2026, 4, 1),
                    date(2026, 4, 1),
                ),
            )
            db_connection.execute(
                """
                INSERT INTO loan_prediction_dates (loan_id, prediction_date, snapshot_source)
                VALUES (%s, %s, 'manual')
                """,
                (loan["loan_id"], date(2026, 3, 15)),
            )

            result = db_connection.execute(
                """
                SELECT * FROM trusted_loan_analytical_dataset
                WHERE loan_id = %s AND prediction_date = %s
                """,
                (loan["loan_id"], date(2026, 3, 15)),
            ).fetchone()

            assert result is not None
            assert result["installments_due"] == 1
            assert result["installments_paid"] == 0
            assert result["installments_overdue"] == 1
            assert result["paid_amount_as_of_date"] == Decimal("0.00")
            assert result["outstanding_amount_as_of_date"] == Decimal("100.00")
            assert result["maximum_days_past_due"] == 42
            assert result["has_30_day_delinquency"] is True
            assert result["has_90_day_delinquency"] is False

            feature_rows = db_connection.execute(OPERATIONAL_SOURCE_QUERY).fetchall()
            feature_source = next(
                row for row in feature_rows if row["loan_id"] == loan["loan_id"]
            )
            assert feature_source["customer_loan_count"] == 1
            assert feature_source["customer_outstanding_debt"] == Decimal("1200.00")
            assert feature_source["customer_monthly_principal_obligation"] == Decimal(
                "100.00"
            )
            assert feature_source["average_days_past_due"] == Decimal("42.0000")
            assert feature_source["late_payment_count"] == 1
            assert feature_source["previous_default_count"] == 0
            assert feature_source["account_activity_count_90d"] == 0
