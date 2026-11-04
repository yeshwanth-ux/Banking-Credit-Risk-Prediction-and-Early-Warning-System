"""End-to-end contracts and important failure scenarios."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg
import pytest

import src.database as database_module
from src.database import DatabaseSettings, wait_for_database
from src.pipeline import validate_source_task


def test_raw_validation_rejects_invalid_input_before_database_write(tmp_path: Path) -> None:
    source = tmp_path / "invalid.csv"
    pd.DataFrame({"unexpected": [1]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="data dictionary"):
        validate_source_task(source)


def test_unavailable_database_fails_after_bounded_attempts(monkeypatch) -> None:  # noqa: ANN001
    def unavailable_connection(*args, **kwargs):  # noqa: ANN002, ANN003
        raise psycopg.OperationalError("simulated unavailable database")

    monkeypatch.setattr(database_module, "connection", unavailable_connection)
    settings = DatabaseSettings(
        host="unavailable.invalid",
        port=5432,
        database="credit_risk",
        user="test",
        password="test",
    )

    with pytest.raises(ConnectionError, match="did not become ready"):
        wait_for_database(settings, attempts=2, delay_seconds=0)
