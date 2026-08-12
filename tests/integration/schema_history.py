"""How to fabricate an older database, honestly (M4-T10, moved M5-T02).

A **module**, not a `conftest.py`. It was a conftest until `tests/gui/` grew
one too, and `from conftest import revert_to` then resolved to whichever
directory pytest had put on `sys.path` first — importing a conftest **by module
name** works only while there is exactly one. Fixtures belong in a conftest;
importable helpers belong in a module with a name of their own.

The helper exists because the same mistake was made three times: a test that fabricates an *older* database by dropping the tables it
happens to know about. Every new migration step then breaks it, and the failure
arrives as `CREATE TABLE … already exists` in a test that has nothing to do with
the new step.

`revert_to` drops everything a later step created, from one table that a new
migration must extend — and `test_the_revert_map_covers_every_step` fails the
moment it does not, which is the point.
"""

from __future__ import annotations

import sqlite3

from nanoscope.infrastructure.storage import SCHEMA_VERSION

#: What each migration step created, so it can be undone. A step that adds a
#: table adds a line here; SQLite drops a table's indexes with it, so only the
#: tables are listed.
TABLES_BY_VERSION: dict[int, tuple[str, ...]] = {
    1: ("images",),
    2: ("analysis_runs", "detections"),
    3: ("annotations",),
    4: ("settings",),
    5: ("models",),
}


def revert_to(conn: sqlite3.Connection, version: int) -> None:
    """Put a database back to how schema `version` left it.

    Not "drop the tables I remember": a database that claims to be version 2
    while carrying a version 4 table is not a version 2 database, and the
    migration mechanism is right to refuse it. This drops everything above the
    target, so the fabrication is honest.
    """
    for step in sorted(TABLES_BY_VERSION, reverse=True):
        if step > version:
            for table in TABLES_BY_VERSION[step]:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"PRAGMA user_version = {version:d}")
    conn.commit()


def test_the_revert_map_covers_every_step() -> None:
    """The guard that makes the next migration's author notice this file."""
    assert max(TABLES_BY_VERSION) == SCHEMA_VERSION
    assert sorted(TABLES_BY_VERSION) == list(range(1, SCHEMA_VERSION + 1))
