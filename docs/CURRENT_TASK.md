# CURRENT TASK

**ID:** `M4-T02`
**Title:** SQLite schema v1 and the forward-migration mechanism
**Milestone:** M4 — Application layer, second task
**Defect:** — (W1: no application layer exists) · **ADR:** **ADR-0039**
**Branch:** `feat/m4-application-layer` — M4 changes no scientific output (PROJECT_RULES §7)
**Status:** **done 2026-08-12.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M4-T01 decided *where* the schema version lives — `PRAGMA user_version`, in the database, separate
from the manifest's `format_version` — and deliberately owned nothing else about the database:

> *`schema_version` … lives in `database.sqlite`, as `PRAGMA user_version`, and M4-T02 owns it.*

Everything above this task reads or writes rows. `ProjectRepository` (M4-T03) needs tables to
implement; `CreateProject` / `ImportImages` (M4-T04) need a database to create and open. If the
schema arrives with the first repository, the migration mechanism arrives *after* the first
tables — which is the one order that cannot work, because the mechanism's first job is creating
them.

The same argument as M4-T01, one layer down: **the operator's data is on the far side of this
too.** A schema without a migration path is a schema that can never change.

---

## The decisions this task has to make

**1. What shape is a migration?** An ordered sequence of steps, each a version number and the
statements that reach it, applied in order, forward only.

| | |
|---|---|
| **An ordered list in code, `SCHEMA_VERSION` derived from its last entry** ✅ | One place to add a step; the constant cannot drift from the list because it *is* the list's last element. A test asserts the versions are contiguous from 1 |
| A constant plus a `dict` of migrations | Two things to update, and the failure mode is a schema that claims a version nothing produces |
| `.sql` files discovered on disk | Ordering by filename, a packaging problem, and a schema that depends on files shipping correctly |

**2. Atomicity.** A migration that fails halfway must leave the database at the version it
started at. Python's `sqlite3` opens an implicit transaction only for DML — **DDL runs in
autocommit** — so `CREATE TABLE` is *not* wrapped for free. Each step therefore runs under an
explicit `BEGIN` / `COMMIT`, with `PRAGMA user_version` set inside the same transaction (it is
stored in the database header and is transactional). A test breaks a step on purpose and asserts
the version did not move.

**3. What is in v1?** Only what has a caller: an `images` table. Adding a table is what the
mechanism in decision 1 exists for, so guessing at `measurements`, `jobs`, `settings` and `logs`
now buys nothing and commits us to columns written before their first reader. Each later task
brings its own migration.

**4. Journal mode: not WAL.** WAL adds `database.sqlite-wal` and `-shm` to a directory whose
layout is a published contract, and a project copied while a WAL is unflushed is ADR-0003's
"half-copied project" with committed data in a file the operator may not have copied. One desktop
writer needs none of what WAL buys.

**5. Foreign keys on, per connection.** SQLite defaults them **off**, per connection, so a
`REFERENCES` clause with no `PRAGMA foreign_keys = ON` is decoration. The pragma belongs in the
connect helper, outside any transaction (it is a silent no-op inside one).

**6. Which error?** `ProjectFormatError`, the same one M4-T01 raised for a newer
`format_version`. To the operator a newer schema is the same sentence — *this is not something I
can open* — and ADR-0038 already made that class carry four cases for exactly that reason.

**7. Constraints the database enforces, not the code.** Two rules from the format contract become
`CHECK` clauses, because a rule enforced by care is a rule already broken somewhere:

- a stored path is relative — ADR-0003's *"the application must be careful never to write an
  absolute path"* becomes a constraint that refuses one
- `modality` is one of the three `Modality` members, pinned by a test that reads the enum, so the
  duplication cannot drift

---

## Scope

**In scope**

1. `nanoscope/infrastructure/storage/database.py` — `SCHEMA_VERSION`, `MIGRATIONS`, `connect`,
   `open_database`, `schema_version`, `migrate`
2. The v1 `images` table
3. **ADR-0039** — the migration mechanism, the v1 scope rule, no WAL, foreign keys on
4. `docs/ProjectFormat.md` §3 — `schema_version` is no longer "owned by M4-T02", it is
   implemented; and what a reader may assume about `database.sqlite`
5. Tests: a fresh database, an idempotent second run, a newer schema refused, a broken step
   rolled back, the pragmas, and both `CHECK` clauses

**Out of scope**

- **The repository** — M4-T03. This task creates and versions the tables; nothing here reads or
  writes a row except the tests that prove the tables work
- **Creating a project directory** — M4-T04
- **Every other table.** By decision 3, and on purpose
- **The integrity check** reconciling rows against files — M4-T03, which needs the repository

---

## Expected blast radius

- **Zero golden differences.** No numerical code is imported, let alone touched. A red golden
  here means something pulled in the science by accident
- One new module, one new document section, one ADR, one new test file
- `sqlite3` is stdlib; no new dependency

---

## Definition of done

- [x] `database.py` — migrations, pragmas, the version check
- [x] The v1 `images` table, with its two `CHECK` clauses
- [x] ADR-0039
- [x] `docs/ProjectFormat.md` updated where it defers to this task
- [x] Tests covering: fresh, idempotent, newer-refused, rollback, pragmas, constraints
- [x] `make check` green — 524 tests, golden byte-identical
- [x] `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [x] Commit: `M4-T02: the schema has a version and a way forward`

---

## What it turned up

**A migration is not atomic for free, and the default that makes it so is Python's, not
SQLite's.** `sqlite3` opens a transaction implicitly before **DML only** — `CREATE TABLE` runs in
autocommit. A step that created two tables and failed on its third statement would therefore leave
those two tables behind **at the old version number**, and the next open would try to create them
again. Nothing about the obvious loop over statements hints at it. The explicit `BEGIN` is the
whole safety property, and the test that breaks a step on purpose is the only thing that proves it.

**`PRAGMA foreign_keys` has two silent failure modes, not one.** It is off by default *and* per
connection, and setting it inside a transaction is a no-op that reports success. Either mistake
leaves every future `REFERENCES` clause as decoration, with a test suite that passes.

**Deciding what v1 contains took longer than writing it.** The pull toward designing
`measurements`, `jobs`, `settings` and `logs` now is strong, and it is the same instinct M2-T08
refused for the ports: an interface written before its first adapter is a guess that gets
rewritten. Here the guess would be on an operator's disk, behind a no-destructive-migrations rule.

**The journal mode turned out to be a format question.** WAL reads as a performance knob until you
notice it adds two files to a directory whose contents this project *published* as a contract one
task ago — and that a project copied mid-write would then have committed data outside it.

---

## Notes

M4's risk profile held for the second time: the golden is byte-identical and nothing numerical is
imported by this module. **M4-T03** takes the tables and puts a repository on them, including the
integrity check ADR-0003 has been owed since it named the dangling row.
