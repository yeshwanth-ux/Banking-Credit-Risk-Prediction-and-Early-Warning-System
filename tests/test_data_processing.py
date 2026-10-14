from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.data_processing import (
    DatasetMetrics,
    QualityCheck,
    clean_identifier,
    clean_optional_text,
    render_quality_report,
)


def test_clean_identifier_is_repeatable() -> None:
    assert clean_identifier("  loan  42 / A ") == "LOAN-42-A"
    assert clean_identifier("LOAN-42-A") == "LOAN-42-A"
    with pytest.raises(ValueError, match="empty"):
        clean_identifier(" /// ")


def test_clean_optional_text_collapses_whitespace() -> None:
    assert clean_optional_text("  salaried   full time ") == "salaried full time"
    assert clean_optional_text("   ") is None
    assert clean_optional_text(None) is None


def test_quality_report_contains_metrics_and_failed_checks() -> None:
    checks = [
        QualityCheck("unique_key", 0, "Rows are unique."),
        QualityCheck("valid_ranges", 2, "Values are in range."),
    ]
    metrics = DatasetMetrics(10, 8, 1000, Decimal("0.30"))

    report = render_quality_report(checks, metrics, date(2026, 10, 14))

    assert "**Overall status:** FAIL" in report
    assert "1/2" in report
    assert "| Operational point-in-time dataset | 10 | 8 | n/a |" in report
    assert "30.00%" in report
    assert "| `valid_ranges` | FAIL | 2 |" in report
