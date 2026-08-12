# ADR-0039 — The schema has a version and a way forward, and only the tables it has readers for

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T02)
- **Affects:** `infrastructure/storage` · M4 · every task that stores a row

## Context

ADR-0038 decided where the schema version lives — `PRAGMA user_version`, in the database, separate
from the manifest's `format_version` — and owned nothing else about the database. ADR-0003 stated
the rule it must satisfy: *"every schema has a version and a forward migration. No destructive
migrations."*

Everything above this task reads or writes rows: the repository (M4-T03), the lifecycle use cases
(M4-T04), measurements (M4-T05), annotations (M4-T07), settings (M4-T10), export (M4-T11), the log
sink (M4-T14). If the schema arrives with the first repository, the migration mechanism arrives
*after* the first tables — which is the one order that cannot work, because creating them is the
mechanism's first job.

Nothing is on disk yet, so this decision is still free. It stops being free the moment one
operator has a project.

## Decision

### 1. A migration is a version and the statements that reach it, in an ordered list

`MIGRATIONS` is a tuple of `(version, statements)` in `database.py`, applied in order, forward
only. `SCHEMA_VERSION` is **derived** from its last entry rather than declared beside it: a
constant that can disagree with the list of migrations eventually does, and the disagreement
produces a database claiming a version nothing produced.

A step that has shipped is never edited. A project on disk has already run it, and rewriting it
makes two databases that both claim the same version and do not have the same tables.

Rejected: `.sql` files discovered on disk (ordering by filename, and a packaging problem — a
schema that depends on files shipping correctly); a `dict` of migrations beside a hand-written
constant (two things to update, and the failure mode is silent).

### 2. Each step is atomic, and that needs an explicit `BEGIN`

Python's `sqlite3` opens a transaction implicitly before **DML only**. DDL runs in autocommit, so
`CREATE TABLE` is not wrapped for free — a step that failed on its third statement would leave two
tables behind at the *old* version number, and the next run would try to create them again.

Each step therefore runs under an explicit `BEGIN` / `COMMIT`, with `PRAGMA user_version` set
inside the same transaction: it is stored in the database header and is transactional, so the
tables and the number that describes them move together or not at all. A test breaks a step on
purpose and asserts the version did not move and no table survived.

### 3. Version 1 holds only tables that have a reader today

One table: `images`. Not `measurements`, `jobs`, `settings` or `logs`, whose owning tasks are not
written.

Adding a table is precisely what decision 1 exists for. Designing four of them now would fix
columns before their first reader exists, and columns invented ahead of a caller are the ones that
turn out to be the wrong shape — at which point they are on an operator's disk and need a
migration to remove, which "no destructive migrations" makes expensive.

This is the same rule M2-T08 applied to the ports ("an interface written before its first adapter
is a guess that gets rewritten"), one layer down and with the operator's data attached.

### 4. No WAL

The default rollback journal, not `journal_mode = WAL`.

WAL adds `database.sqlite-wal` and `database.sqlite-shm` to a directory whose layout is a
**published contract** (`docs/ProjectFormat.md` §1), and a project copied while a WAL is unflushed
is ADR-0003's known "half-copied project" hazard with a sharper edge: committed transactions live
in a file the operator did not know to copy. What WAL buys is concurrent readers during a write,
and this is one desktop writer.

### 5. Foreign keys are enabled per connection, in the connect helper

SQLite defaults `foreign_keys` to **off**, and the setting is per connection, not per database. A
`REFERENCES` clause written by a later table without this pragma is decoration. It is also a
silent no-op inside a transaction, so it is set the moment the connection exists and before
anything else runs.

### 6. Two rules become `CHECK` clauses instead of conventions

- **`relative_path NOT LIKE '/%'`.** ADR-0003 asks the application to "be careful never to write
  an absolute path". Care is not a mechanism; the constraint refuses one, in the one place every
  writer passes through.
- **`modality IN ('afm', 'sem', 'tem')`.** A copy of `core.values.Modality` in SQL, which is a
  drift risk — so the test that proves the constraint iterates the enum, and the copy cannot go
  stale without a red test.

### 7. A newer schema raises `ProjectFormatError`

The same class M4-T01 raises for a newer `format_version`, naming both versions. To the operator
it is the same sentence — *this is not something I can open* — and ADR-0038 already gave that
class four cases for exactly that reason. A second error type would be a distinction only the
implementation cares about.

## Consequences

**Positive**

- The schema can change, which is the property that is impossible to add later.
- A half-applied migration is not a state this code can produce, and a test proves it rather than
  the docstring claiming it.
- A project directory contains the files `ProjectFormat.md` says it contains, at all times.
- The relative-path invariant is enforced by the database, so no future repository has to remember
  it.
- v1 is small enough to read in one sitting, and every column in it has a caller.

**Negative**

- One table is not much of a "schema v1", and each later task now carries a migration it would
  otherwise not have written. Accepted: that migration is three lines and a test, against columns
  designed without a reader.
- No WAL means a writer blocks readers. Irrelevant at one desktop user, and revisitable — a
  journal mode is a pragma, not a format change, though the layout note in `ProjectFormat.md`
  would need revisiting with it.
- The `modality` values exist in two languages. Mitigated by the test, not by a comment.

**Neutral**

- `sqlite3` is stdlib; this adds no dependency.
- `SCHEMA_VERSION` starts at 1 with nothing to migrate *from* except an empty file. The first
  interesting migration is written by whichever task adds the second table.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A migration framework (alembic, yoyo) | A dependency, a config file and a CLI for what is a list of statements and a `PRAGMA` — and it would own the database an offline desktop application ships inside a user's directory |
| One `CREATE TABLE IF NOT EXISTS` block run at every open | Cannot express a change: adding a column to an existing table is exactly what it does not do, and it silently accepts a database it did not create |
| A `schema_version` row in a `meta` table | A table to read before you know the schema, where SQLite already provides a header field that costs no query and is transactional |
| Design all of M4's tables now | Columns fixed before their readers exist, and "no destructive migrations" makes removing a wrong one expensive |
| WAL for speed | Two extra files in a documented layout, and committed data outside a copied project directory |
| Enforce relative paths in the repository only | One code path today, and one more with every writer; the constraint holds for all of them, including a script the operator writes |

## Compliance

- `tests/unit/test_database.py` covers: a fresh database, an idempotent second run, a newer schema
  refused by both versions, a failed step leaving nothing behind, both pragmas, both `CHECK`
  clauses, and that the project directory holds only `database.sqlite`.
- `SCHEMA_VERSION` is never written as a literal; it is `MIGRATIONS[-1][0]`, and a test asserts the
  versions are contiguous from 1.
- No shipped migration step is edited. A change to the schema is a new step.
- No column holds image or mask bytes (ADR-0003), and no stored path is absolute.

## References

- ADR-0003 (projects are directories; SQLite stores metadata only) — the rule this implements
- ADR-0038 (the project format is a versioned contract) — decided where `schema_version` lives
- ADR-0019 / ADR-0025 (an unknown scale is a state) — why `pixel_size_nm` is nullable
- `docs/ProjectFormat.md` §3 · `docs/TASKS.md` M4-T02, M4-T03
