"""Apply ordered, checksum-protected PostgreSQL migrations."""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from psycopg import Connection

from src.database import DatabaseSettings, connection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIGRATION_DIR = PROJECT_ROOT / "sql" / "migrations"
MIGRATION_PATTERN = re.compile(r"^V(?P<version>\d{3})__(?P<name>[a-z0-9_]+)\.sql$")
MIGRATION_LOCK_ID = 1_903_117_541


@dataclass(frozen=True, order=True)
class Migration:
    version: int
    name: str
    path: Path
    checksum: str


def discover_migrations(directory: Path = DEFAULT_MIGRATION_DIR) -> list[Migration]:
    migrations: list[Migration] = []
    versions: set[int] = set()
    for path in sorted(directory.glob("V*.sql")):
        match = MIGRATION_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Invalid migration filename: {path.name}")
        version = int(match.group("version"))
        if version in versions:
            raise ValueError(f"Duplicate migration version: {version:03d}")
        versions.add(version)
        migrations.append(
            Migration(
                version=version,
                name=match.group("name"),
                path=path,
                checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    if not migrations:
        raise ValueError(f"No migrations found in {directory}")
    return migrations


def ensure_history_table(db_connection: Connection) -> None:
    db_connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            checksum CHAR(64) NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def applied_migrations(db_connection: Connection) -> dict[int, str]:
    rows = db_connection.execute(
        "SELECT version, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()
    return {row["version"]: row["checksum"] for row in rows}


def apply_migrations(
    settings: DatabaseSettings | None = None,
    directory: Path = DEFAULT_MIGRATION_DIR,
) -> list[Migration]:
    migrations = discover_migrations(directory)
    applied_now: list[Migration] = []

    with connection(settings) as db_connection:
        db_connection.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
        ensure_history_table(db_connection)
        applied = applied_migrations(db_connection)

        for migration in migrations:
            existing_checksum = applied.get(migration.version)
            if existing_checksum:
                if existing_checksum != migration.checksum:
                    raise RuntimeError(
                        f"Applied migration V{migration.version:03d} checksum changed"
                    )
                continue

            db_connection.execute(migration.path.read_text(encoding="utf-8"))
            db_connection.execute(
                """
                INSERT INTO schema_migrations (version, name, checksum)
                VALUES (%s, %s, %s)
                """,
                (migration.version, migration.name, migration.checksum),
            )
            applied_now.append(migration)

    return applied_now


def migration_status(
    settings: DatabaseSettings | None = None,
    directory: Path = DEFAULT_MIGRATION_DIR,
) -> list[tuple[Migration, bool]]:
    migrations = discover_migrations(directory)
    with connection(settings) as db_connection:
        ensure_history_table(db_connection)
        applied = applied_migrations(db_connection)
    return [(migration, migration.version in applied) for migration in migrations]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("up", "status"), nargs="?", default="up")
    args = parser.parse_args()

    if args.command == "up":
        applied = apply_migrations()
        if applied:
            for migration in applied:
                print(f"Applied V{migration.version:03d} {migration.name}")
        else:
            print("Database schema is already current")
        return

    for migration, is_applied in migration_status():
        state = "applied" if is_applied else "pending"
        print(f"V{migration.version:03d} {migration.name}: {state}")


if __name__ == "__main__":
    main()
