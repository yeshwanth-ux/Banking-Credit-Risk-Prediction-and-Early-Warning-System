from __future__ import annotations

from pathlib import Path

import pytest

from src.migrations import discover_migrations


def test_discovers_versioned_migrations_in_order(tmp_path: Path) -> None:
    (tmp_path / "V002__second.sql").write_text("SELECT 2;", encoding="utf-8")
    (tmp_path / "V001__first.sql").write_text("SELECT 1;", encoding="utf-8")

    migrations = discover_migrations(tmp_path)

    assert [migration.version for migration in migrations] == [1, 2]
    assert [migration.name for migration in migrations] == ["first", "second"]
    assert all(len(migration.checksum) == 64 for migration in migrations)


def test_rejects_duplicate_migration_versions(tmp_path: Path) -> None:
    (tmp_path / "V001__first.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "V001__duplicate.sql").write_text("SELECT 2;", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate migration version"):
        discover_migrations(tmp_path)


def test_rejects_invalid_migration_filename(tmp_path: Path) -> None:
    (tmp_path / "V1__invalid.sql").write_text("SELECT 1;", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid migration filename"):
        discover_migrations(tmp_path)
