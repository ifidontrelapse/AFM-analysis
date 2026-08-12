"""The schema's version and the mechanism that moves it (M4-T02, ADR-0039).

Three things are pinned here, and they fail for different reasons:

1. **The mechanism.** A fresh database ends at `SCHEMA_VERSION`, a second run
   does nothing, a newer schema is refused, and a step that fails leaves the
   version where it was — which is the only property that makes a migration
   safe to ship.
2. **The pragmas.** Foreign keys are off by default in SQLite, per connection,
   and no WAL means a project directory holds the files the format contract
   says it holds.
3. **The v1 table**, including the two `CHECK` clauses that enforce rules the
   code would otherwise have to remember.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from nanoscope.core.errors import NanoscopeError, ProjectFormatError
from nanoscope.core.values import Modality
from nanoscope.infrastructure.storage import (
    DATABASE_NAME,
    MIGRATIONS,
    SCHEMA_VERSION,
    connect,
    migrate,
    open_database,
    schema_version,
)

_ROW = (
    "INSERT INTO images "
    "(relative_path, display_name, modality, sha256, pixel_size_nm, imported_utc) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    """An empty database, in memory, with the project's pragmas."""
    conn = connect(":memory:")
    yield conn
    conn.close()


class TestTheMigrationMechanism:
    def test_an_empty_database_is_version_zero(self, db: sqlite3.Connection) -> None:
        """Version 0 is not a special case in the list: it is what a file with
        no tables reports, and every table in existence was created by a step."""
        assert schema_version(db) == 0

    def test_migrating_reaches_the_current_version(self, db: sqlite3.Connection) -> None:
        assert migrate(db) == (0, SCHEMA_VERSION)
        assert schema_version(db) == SCHEMA_VERSION

    def test_migrating_again_does_nothing(self, db: sqlite3.Connection) -> None:
        migrate(db)

        assert migrate(db) == (SCHEMA_VERSION, SCHEMA_VERSION)
        assert schema_version(db) == SCHEMA_VERSION

    def test_a_newer_schema_is_refused_and_names_both_versions(
        self, db: sqlite3.Connection
    ) -> None:
        """The same rule as the manifest's, one layer down: a forward migration
        cannot be written by the past (ADR-0038)."""
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

        with pytest.raises(ProjectFormatError) as excinfo:
            migrate(db)

        message = str(excinfo.value)
        assert str(SCHEMA_VERSION + 1) in message
        assert str(SCHEMA_VERSION) in message

    def test_the_refusal_is_catchable_as_ours(self, db: sqlite3.Connection) -> None:
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

        with pytest.raises(NanoscopeError):
            migrate(db)

    def test_a_step_that_fails_moves_nothing(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property the explicit `BEGIN` buys. Python's `sqlite3` opens a
        transaction for DML only, so without it a half-applied `CREATE TABLE`
        would survive at the old version — a schema that lies about itself."""
        broken = ((1, ("CREATE TABLE fine (id INTEGER PRIMARY KEY)", "NOT SQL AT ALL")),)
        monkeypatch.setattr("nanoscope.infrastructure.storage.database.MIGRATIONS", broken)

        with pytest.raises(sqlite3.Error):
            migrate(db)

        assert schema_version(db) == 0
        assert _tables(db) == []

    def test_the_versions_are_contiguous_from_one(self) -> None:
        """A gap would leave a database at a version no step targets, which is a
        database that can never be migrated again."""
        assert [version for version, _ in MIGRATIONS] == list(range(1, len(MIGRATIONS) + 1))

    def test_the_constant_is_the_last_step(self) -> None:
        """Derived, not declared beside the list — a constant that can disagree
        with the migrations eventually does."""
        assert MIGRATIONS[-1][0] == SCHEMA_VERSION


class TestOpeningAProjectDatabase:
    def test_it_is_created_where_the_format_says(self, tmp_path: Path) -> None:
        open_database(tmp_path).close()

        assert (tmp_path / DATABASE_NAME).exists()

    def test_it_opens_migrated(self, tmp_path: Path) -> None:
        conn = open_database(tmp_path)

        assert schema_version(conn) == SCHEMA_VERSION
        assert "images" in _tables(conn)
        conn.close()

    def test_reopening_an_existing_database_keeps_its_rows(self, tmp_path: Path) -> None:
        conn = open_database(tmp_path)
        conn.execute(_ROW, ("images/a.spm", "a.spm", "afm", "0" * 64, 1.95, "2026-08-12T00:00:00Z"))
        conn.commit()
        conn.close()

        conn = open_database(tmp_path)
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        conn.close()

    def test_a_newer_database_is_refused_without_leaking_the_connection(
        self, tmp_path: Path
    ) -> None:
        conn = open_database(tmp_path)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        conn.commit()
        conn.close()

        with pytest.raises(ProjectFormatError):
            open_database(tmp_path)

    def test_the_directory_holds_only_the_file_the_contract_names(self, tmp_path: Path) -> None:
        """No WAL: `-wal` and `-shm` would be two files in a published layout,
        and a project copied mid-write would leave committed data behind."""
        conn = open_database(tmp_path)
        conn.execute(_ROW, ("images/a.spm", "a.spm", "afm", "0" * 64, None, "2026-08-12T00:00:00Z"))
        conn.commit()

        assert sorted(p.name for p in tmp_path.iterdir()) == [DATABASE_NAME]
        conn.close()


class TestThePragmas:
    def test_foreign_keys_are_on(self, db: sqlite3.Connection) -> None:
        """SQLite defaults them off, per connection. Without this pragma every
        `REFERENCES` clause a later table writes is decoration."""
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_rows_are_readable_by_column_name(self, tmp_path: Path) -> None:
        conn = open_database(tmp_path)
        conn.execute(_ROW, ("images/a.spm", "a.spm", "afm", "0" * 64, 1.95, "2026-08-12T00:00:00Z"))

        row = conn.execute("SELECT * FROM images").fetchone()

        assert row["relative_path"] == "images/a.spm"
        conn.close()


class TestTheImagesTable:
    def test_a_row_round_trips(self, db: sqlite3.Connection) -> None:
        migrate(db)
        db.execute(_ROW, ("images/a.spm", "a.spm", "afm", "0" * 64, 1.95, "2026-08-12T00:00:00Z"))

        row = db.execute("SELECT * FROM images").fetchone()

        assert (row["display_name"], row["modality"], row["pixel_size_nm"]) == (
            "a.spm",
            "afm",
            1.95,
        )

    def test_an_unknown_pixel_scale_is_null(self, db: sqlite3.Connection) -> None:
        """A state, not a fabricated 1.0 — the same invariant the entities carry
        (ADR-0019, ADR-0025)."""
        migrate(db)
        db.execute(_ROW, ("images/a.npy", "a.npy", "afm", "0" * 64, None, "2026-08-12T00:00:00Z"))

        assert db.execute("SELECT pixel_size_nm FROM images").fetchone()[0] is None

    def test_an_absolute_path_is_refused(self, db: sqlite3.Connection) -> None:
        """ADR-0003 asks the application to "be careful never to write an
        absolute path". Care is not a mechanism; a CHECK is."""
        migrate(db)

        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                _ROW,
                ("/home/op/a.spm", "a.spm", "afm", "0" * 64, 1.95, "2026-08-12T00:00:00Z"),
            )

    def test_the_same_file_cannot_be_imported_twice(self, db: sqlite3.Connection) -> None:
        migrate(db)
        db.execute(_ROW, ("images/a.spm", "a.spm", "afm", "0" * 64, 1.95, "2026-08-12T00:00:00Z"))

        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                _ROW, ("images/a.spm", "copy", "afm", "1" * 64, 1.95, "2026-08-12T00:00:00Z")
            )

    @pytest.mark.parametrize("modality", list(Modality))
    def test_every_modality_the_code_knows_is_accepted(
        self, db: sqlite3.Connection, modality: Modality
    ) -> None:
        """The CHECK clause names the three values in SQL, which is a copy of
        the enum. This is the test that stops the copy from drifting."""
        migrate(db)

        db.execute(
            _ROW,
            (f"images/{modality}.tif", "x", str(modality), "0" * 64, None, "2026-08-12T00:00:00Z"),
        )

    def test_a_modality_the_code_does_not_know_is_refused(self, db: sqlite3.Connection) -> None:
        migrate(db)

        with pytest.raises(sqlite3.IntegrityError):
            db.execute(_ROW, ("images/a.tif", "a", "stm", "0" * 64, None, "2026-08-12T00:00:00Z"))


def _tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    return [row[0] for row in rows]
