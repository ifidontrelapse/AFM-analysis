"""The project database: its schema, its version, and its way forward (M4-T02).

`project_format.py` owns the directory; this module owns the one file in it that
is not a plain document. The split is ADR-0038's: `format_version` describes the
layout and lives in the manifest, `schema_version` describes the tables and lives
here, as SQLite's own `PRAGMA user_version`. They are bumped for different
reasons.

A schema without a migration path is a schema that can never change, so the
mechanism comes first and the tables come *through* it: version 0 is an empty
file, and every table in existence was created by a migration. That is also why
`MIGRATIONS` holds so little — a table with no caller today is a set of columns
designed before its first reader, and adding one later is exactly what this
mechanism is for (ADR-0039).

What is *not* here: reading and writing rows (M4-T03's repository), creating a
project directory (M4-T04), and every table whose owner has not been written yet.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from nanoscope.core.errors import ProjectFormatError
from nanoscope.infrastructure.storage.project_format import DATABASE_NAME

# The images index. Metadata only: the pixels stay in `images/` as the file the
# operator imported, and the row points at them (ADR-0003).
#
# `relative_path` is relative to the *project root*, not to `images/`, so one
# column answers "where is this file" for every directory in the layout. The
# CHECK is ADR-0003's "the application must be careful never to write an absolute
# path" turned into something that does not rely on care.
#
# `pixel_size_nm` is nullable because an unknown scale is a state, not a
# fabricated 1.0 (ADR-0019, ADR-0025) — the same invariant the entities carry.
_V1 = (
    """
    CREATE TABLE images (
        id            INTEGER PRIMARY KEY,
        relative_path TEXT    NOT NULL UNIQUE,
        display_name  TEXT    NOT NULL,
        modality      TEXT    NOT NULL,
        sha256        TEXT    NOT NULL,
        pixel_size_nm REAL,
        imported_utc  TEXT    NOT NULL,
        CHECK (relative_path NOT LIKE '/%'),
        CHECK (modality IN ('afm', 'sem', 'tem'))
    )
    """,
)

#: Every step from an empty file to the current schema, in order. A step is its
#: target version and the statements that reach it; they run in one transaction
#: and the version moves with them.
#:
#: Adding a step is the only way the schema changes. Never edit a step that has
#: shipped — a project on disk has already run it, and rewriting it makes two
#: databases that both claim the same version.
MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = ((1, _V1),)

#: What this application writes and can read. Derived from the list rather than
#: declared beside it, because a constant that can disagree with the migrations
#: eventually does.
SCHEMA_VERSION = MIGRATIONS[-1][0]


def connect(database_path: Path | str) -> sqlite3.Connection:
    """Open `database_path`, with the pragmas this project's tables assume.

    Does **not** migrate: a repair tool has to be able to look at a database it
    is not allowed to open. `open_database` is the one that migrates.

    Foreign keys are enabled here and not in the schema, because SQLite defaults
    them **off**, per connection — a `REFERENCES` clause without this pragma is
    decoration. It is also a silent no-op inside a transaction, which is why it
    is set the moment the connection exists.
    """
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    """The version of the tables in `conn`. `0` for a database with none."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def migrate(conn: sqlite3.Connection) -> tuple[int, int]:
    """Bring `conn` up to `SCHEMA_VERSION`, and say where it came from.

    Forward-only and never destructive (ADR-0003): a step may add, and may
    rewrite what it fully understands, but it may not discard what it does not
    recognise. Running this on a database already at the current version does
    nothing.

    Returns:
        The version before and after. Equal when there was nothing to do.

    Raises:
        ProjectFormatError: the database declares a schema newer than this
            application knows. Refused rather than migrated, for ADR-0038's
            reason: a forward migration cannot be written by the past.
    """
    was = schema_version(conn)
    if was > SCHEMA_VERSION:
        raise ProjectFormatError(
            f"database schema version {was} is newer than this application "
            f"understands (version {SCHEMA_VERSION}); upgrade nanoscope to open it"
        )

    for version, statements in MIGRATIONS:
        if version > was:
            _apply(conn, version, statements)
    return was, SCHEMA_VERSION


def open_database(project_dir: Path | str) -> sqlite3.Connection:
    """Open a project's `database.sqlite`, migrated and ready. Creates it if absent.

    An absent database is an empty one at version 0, which the migrations then
    fill — so a project whose index was deleted comes back with its tables, and
    with nothing in them. The files under `images/` are what survived, and
    reconciling the two is the repository's integrity check (M4-T03).
    """
    conn = connect(Path(project_dir) / DATABASE_NAME)
    try:
        migrate(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _apply(conn: sqlite3.Connection, version: int, statements: tuple[str, ...]) -> None:
    """Run one migration step, all of it or none of it.

    The explicit `BEGIN` is the point. Python's `sqlite3` opens a transaction
    implicitly before DML only, so `CREATE TABLE` would otherwise run in
    autocommit and a step that failed halfway would leave half a schema behind
    at the old version number — the one state a migration must never produce.

    `PRAGMA user_version` cannot be parameterised, so the value is formatted in;
    it is an `int` from `MIGRATIONS` above, and `:d` is what keeps that true.
    """
    conn.execute("BEGIN")
    try:
        for statement in statements:
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {version:d}")
    except Exception:
        conn.rollback()
        raise
    conn.commit()
