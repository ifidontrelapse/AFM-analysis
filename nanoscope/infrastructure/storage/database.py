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

# What an analysis found, and where the table it produced was written (M4-T05).
#
# The split ADR-0042 decided: the *index* is here — which run happened, on what,
# with which detector, and every detection it produced — and the measurement
# **table** is a file under `results/`, because ADR-0031 made that table variable
# by construction (a core plus blocks, `method` naming the producer), and a
# relational shape for it is either wide with NULLs or an EAV pivot.
#
# `ON DELETE CASCADE` is what makes M4-T02's `PRAGMA foreign_keys = ON`
# load-bearing rather than a precaution: a detection of a particle in a scan the
# project no longer knows about is litter, and unlike the image row it is the
# *derived* half of the pair, so ADR-0040's argument for keeping it does not
# apply.
_V2 = (
    """
    CREATE TABLE analysis_runs (
        id                INTEGER PRIMARY KEY,
        image_id          INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        detector          TEXT    NOT NULL,
        mode              TEXT    NOT NULL,
        modality          TEXT    NOT NULL,
        pixel_size_nm     REAL,
        measurements_path TEXT,
        created_utc       TEXT    NOT NULL,
        CHECK (measurements_path IS NULL OR measurements_path NOT LIKE '/%'),
        CHECK (modality IN ('afm', 'sem', 'tem'))
    )
    """,
    """
    CREATE TABLE detections (
        id         INTEGER PRIMARY KEY,
        run_id     INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
        ordinal    INTEGER NOT NULL,
        x_px       REAL    NOT NULL,
        y_px       REAL    NOT NULL,
        radius_px  REAL    NOT NULL,
        radius_nm  REAL,
        confidence REAL,
        bbox_x1    INTEGER,
        bbox_y1    INTEGER,
        bbox_x2    INTEGER,
        bbox_y2    INTEGER
    )
    """,
    "CREATE INDEX detections_by_run ON detections(run_id)",
    "CREATE INDEX analysis_runs_by_image ON analysis_runs(image_id)",
)

# What the operator drew (M4-T07).
#
# A table rather than the JSON documents ADR-0003's layout imagined, and the
# contrast with `measurements_path` one migration above is the whole rule: a
# measurement table's columns depend on its producer and can be recomputed, an
# annotation is a fixed handful of numbers, edited one at a time with undo behind
# it, and **irreplaceable** (ADR-0044).
#
# `source` is not decoration. Training a model on boxes copied from that model's
# own output is self-confirmation, and a training set that cannot tell hand-drawn
# from adopted cannot avoid it.
_V3 = (
    """
    CREATE TABLE annotations (
        id          INTEGER PRIMARY KEY,
        image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        label       TEXT    NOT NULL,
        x1          REAL    NOT NULL,
        y1          REAL    NOT NULL,
        x2          REAL    NOT NULL,
        y2          REAL    NOT NULL,
        source      TEXT    NOT NULL,
        note        TEXT,
        created_utc TEXT    NOT NULL,
        updated_utc TEXT    NOT NULL,
        CHECK (source IN ('manual', 'from_detection')),
        CHECK (x2 > x1 AND y2 > y1)
    )
    """,
    "CREATE INDEX annotations_by_image ON annotations(image_id)",
)

# The preferences that belong to this project rather than to the operator
# (M4-T10). A key/value table, and `value` is JSON text so a boolean comes back
# a boolean — a settings store that returns everything as a string makes every
# reader parse, and one of them gets it wrong (ADR-0047).
_V4 = (
    """
    CREATE TABLE settings (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,
        updated_utc TEXT NOT NULL
    )
    """,
)

# The models a project can use (M4-T13). W10's replacement: a record with a
# version, a checksum and a provenance, instead of `"./checkpoints/best12x.pt"`
# in a default argument.
#
# `path` may be absolute here, unlike every other path in this schema, and the
# CHECK that forbids one elsewhere is deliberately absent: nobody copies a
# 137 MB checkpoint into every project, so a shared file is the normal case —
# with the consequence that such a project opens on another machine and the
# model is unavailable there (ADR-0050).
_V5 = (
    """
    CREATE TABLE models (
        model_id       TEXT PRIMARY KEY,
        task           TEXT NOT NULL,
        framework      TEXT NOT NULL,
        path           TEXT NOT NULL,
        input_size_px  INTEGER,
        class_map      TEXT NOT NULL,
        provenance     TEXT NOT NULL,
        sha256         TEXT,
        registered_utc TEXT NOT NULL,
        CHECK (task IN ('detect', 'segment')),
        CHECK (framework IN ('ultralytics', 'sam2'))
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
# An annotation may keep an outline (M7-T03, ADR-0072). One nullable column,
# beside the box rather than instead of it: `points IS NULL` is a box drawn as a
# box — every row written before this migration — and a polygon's `x1…y2` are its
# bounding box, so every reader that consumes boxes keeps working unchanged.
#
# Added empty. Nothing migrates, because "no outline" is exactly what the
# existing rows mean.
_V6: tuple[str, ...] = ("ALTER TABLE annotations ADD COLUMN points TEXT",)

# A painted mask lives in a file and the row points at it (M7-T04, ADR-0073).
# PROJECT_RULES §5: no mask bitmaps in the database. Nullable and added empty,
# like v6 — every row written before this painted nothing.
_V7: tuple[str, ...] = ("ALTER TABLE annotations ADD COLUMN mask_path TEXT",)

# What an operator measured by hand (M7-T05, ADR-0074). A **new table**, because
# a line has no area and ADR-0044's annotation shapes are refused without one —
# and a new *word*, because `measurements.csv` already names what an analysis run
# produces. `kind` carries the two tools that share this geometry: a distance and
# a profile (M7-T06).
_V8: tuple[str, ...] = (
    """
    CREATE TABLE rulers (
        id          INTEGER PRIMARY KEY,
        image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        kind        TEXT    NOT NULL,
        x1          REAL    NOT NULL,
        y1          REAL    NOT NULL,
        x2          REAL    NOT NULL,
        y2          REAL    NOT NULL,
        label       TEXT    NOT NULL,
        created_utc TEXT    NOT NULL,
        CHECK (kind IN ('distance', 'profile'))
    )
    """,
    "CREATE INDEX rulers_by_image ON rulers(image_id)",
)

MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, _V1),
    (2, _V2),
    (3, _V3),
    (4, _V4),
    (5, _V5),
    (6, _V6),
    (7, _V7),
    (8, _V8),
)

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

    `check_same_thread=False` because a project opened on the main thread is used
    by jobs on worker threads (M4-T06), and Python's default refuses that
    outright — the connection remembers which thread made it. SQLite's own C
    library is compiled *serialized* in CPython's build (`sqlite3.threadsafety`
    is 3), so sharing the connection is safe at that level; what is **not** safe
    is two threads interleaving the statements of one logical write, and
    `SqliteProjectRepository` holds a lock for that (ADR-0043).
    """
    conn = sqlite3.connect(database_path, check_same_thread=False)
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
