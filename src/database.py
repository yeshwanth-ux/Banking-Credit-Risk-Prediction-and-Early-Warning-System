"""PostgreSQL configuration and connection helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg import Connection, sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    database: str
    user: str
    password: str
    sslmode: str = "prefer"
    connect_timeout: int = 10
    application_name: str = "banking_credit_risk"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> DatabaseSettings:
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        required = {
            "host": os.getenv("POSTGRES_HOST"),
            "database": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing PostgreSQL settings: {', '.join(sorted(missing))}")

        return cls(
            host=str(required["host"]),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=str(required["database"]),
            user=str(required["user"]),
            password=str(required["password"]),
            sslmode=os.getenv("POSTGRES_SSLMODE", "prefer"),
            connect_timeout=int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10")),
            application_name=os.getenv("POSTGRES_APPLICATION_NAME", "banking_credit_risk"),
        )

    def conninfo(self, *, database: str | None = None) -> str:
        return make_conninfo(
            host=self.host,
            port=self.port,
            dbname=database or self.database,
            user=self.user,
            password=self.password,
            sslmode=self.sslmode,
            connect_timeout=self.connect_timeout,
            application_name=self.application_name,
        )


@contextmanager
def connection(
    settings: DatabaseSettings | None = None,
    *,
    autocommit: bool = False,
) -> Generator[Connection[dict[str, Any]], None, None]:
    """Yield a dictionary-row connection and close it deterministically."""

    config = settings or DatabaseSettings.from_env()
    with psycopg.connect(
        config.conninfo(),
        autocommit=autocommit,
        row_factory=dict_row,
    ) as db_connection:
        yield db_connection


def wait_for_database(
    settings: DatabaseSettings | None = None,
    *,
    attempts: int = 10,
    delay_seconds: float = 1.0,
) -> None:
    """Wait until PostgreSQL accepts a simple query or raise the last error."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    last_error: psycopg.Error | None = None
    for attempt in range(attempts):
        try:
            with connection(settings) as db_connection:
                db_connection.execute("SELECT 1")
            return
        except psycopg.Error as error:
            last_error = error
            if attempt < attempts - 1:
                time.sleep(delay_seconds)

    raise ConnectionError("PostgreSQL did not become ready in time") from last_error


def fetch_all(
    statement: sql.Composable | str,
    parameters: Sequence[Any] | None = None,
    settings: DatabaseSettings | None = None,
) -> list[dict[str, Any]]:
    """Execute a read query and return dictionary rows."""

    with connection(settings) as db_connection:
        return list(db_connection.execute(statement, parameters).fetchall())
