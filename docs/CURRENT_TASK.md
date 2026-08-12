# CURRENT TASK

**ID:** `M4-T03`
**Title:** The `ProjectRepository`, and the integrity check ADR-0003 has been owed
**Milestone:** M4 — Application layer, third task
**Defect:** — (W1: no application layer exists) · **ADR:** **ADR-0040** (to be written)
**Branch:** `feat/m4-application-layer` — M4 changes no scientific output (PROJECT_RULES §7)
**Status:** planned 2026-08-12, implementation next.

---

## Why this task is next

M4-T01 decided what a project directory is; M4-T02 gave it tables. Nothing yet puts a row in one
or finds the file it names — and two things are waiting on that:

1. **M4-T04's use cases.** `ImportImages` and `ListImages` are a repository with a name on top.
2. **ADR-0003's unpaid debt**, written down as a *known cost* of the design and never collected:

> *Two sources of truth to keep consistent … Deleting a file behind the application's back
> produces a dangling row; the repository layer must reconcile (a startup integrity check is
> required).*

That check has no owner until this task. It is also the piece most likely to be quietly skipped,
because everything works in a test where nobody deletes anything.

---

## The decisions this task has to make

**1. Does the port arrive now?** Yes — and `core/ports/__init__.py` already committed to when:
*"the rest ship with their first adapter"*, with `ProjectRepository` listed as arriving with "the
SQLite schema and its repository". That is this task. The Task column on that row says **M6**,
which was true when the table was written and is now wrong; the code decides and the document gets
fixed (PROJECT_RULES §8).

The port is not decoration here. M4-T04's use cases live in `application/`, which may import
`core` and nothing else — typing a use case against the SQLite class would put `infrastructure` on
the application's import list, which Architecture §3.2 forbids outright.

**2. Who computes the checksum?** The repository, from the file the row points at — never a
caller passing one in. A checksum accepted as an argument is a checksum that can describe a
different file, and the only thing it would then prove is that two callers agreed. One funnel, and
it makes `add_image` require that the file is already inside the project.

**3. What does the integrity check do about what it finds?** It **reports**. It does not delete a
row whose file is absent, and it does not import a file that has no row.

| | |
|---|---|
| **Report both directions, change nothing** ✅ | A missing file is as likely to be an unmounted drive or a half-finished copy as a deletion, and the row carries measurements the file does not. ADR-0003 forbids destructive migrations; deleting user data on *open* would be worse than one |
| Delete dangling rows automatically | Silent data loss triggered by a mount point |
| Import untracked files automatically | Guesses that a file under `images/` was meant to be in the project, and invents its modality |

**4. Are contents verified on open?** No. Comparing every checksum means reading every scan on
every open — the `data/` directory here holds 628 SPM files. The check compares *existence*, which
is a `stat` per row; the checksum stays in the row for when an operator asks a question that needs
it. Stated in the ADR so the omission is a decision rather than an oversight.

**5. Relative paths: enforced where?** In the repository, which is the one funnel every writer
passes through — it takes a path, refuses one outside the project root, and stores it relative and
POSIX-separated. M4-T02's `CHECK` stays as the backstop that catches a writer who does not use the
repository. Belt and braces on purpose: the constraint is what makes the rule true of the
*database*, the repository is what makes the error message useful.

**6. What comes back from a query?** `ImageRecord`, an entity — never a `sqlite3.Row`. A Row is
untyped, and returning one would put `sqlite3` in the vocabulary of every layer above.

---

## Scope

**In scope**

1. `core/entities/project.py` — `ImageRecord` and `IntegrityReport`
2. `core/ports/project_repository.py` — the `ProjectRepository` Protocol, and the ports table
   corrected
3. `infrastructure/storage/project_repository.py` — `SqliteProjectRepository`: `open`, `close`,
   context-manager use, `add_image`, `get_image`, `list_images`, `remove_image`, `check_integrity`
4. **ADR-0040** — report-don't-reconcile, the checksum funnel, the port's arrival
5. Integration tests in a new `tests/integration/`: a real directory, a real database, files
   deleted behind the application's back, and a project **moved to another directory** — which is
   the test ADR-0003 asks for by name

**Out of scope**

- **Creating a project directory** — `CreateProject` is M4-T04. The tests build a directory with
  `new_manifest` + `mkdir`, three lines, so this task does not quietly own the lifecycle
- **Copying an imported file into `images/`** — also M4-T04. This task records a file that is
  already there
- **Every other table.** Measurements and annotations arrive with their tasks, each with a
  migration (ADR-0039)
- **Resolving what the integrity check finds.** Reporting is the contract; acting on a report is a
  use case with an operator behind it

---

## Expected blast radius

- **Zero golden differences.** The golden enumerates the fields of `PipelineConfig`,
  `PipelineResult` and `Detection`; a new entity module is invisible to it
- Two new modules, one new entity module, one ADR, one new test directory
- No new dependency

---

## Definition of done

- [ ] `ImageRecord`, `IntegrityReport`, and the `ProjectRepository` port
- [ ] `SqliteProjectRepository`, with paths stored relative and checksums computed in one place
- [ ] `check_integrity` reporting both directions and changing nothing
- [ ] ADR-0040, and the corrected row in the ports table
- [ ] Integration tests, including a moved project and a file deleted behind our back
- [ ] `make check` green, golden byte-identical
- [ ] `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [ ] Commit: `M4-T03: the repository reports what it finds, and never reconciles by deleting`
