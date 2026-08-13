# CURRENT TASK

**ID:** `M6-T07`
**Title:** The export an operator can ask for, and the scope they mean
**Milestone:** M6 — Analysis workflow in the GUI, seventh task
**Defect:** — · **ADR:** **ADR-0067**
**Branch:** `feat/m6-analysis-workflow`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M6's first exit criterion is *load → detect → segment → measure → **export CSV**, entirely through
the UI*, and the export is the last step of it. `export_measurements` has been in `application` since
M4-T11 with tests as its only callers: the operator's way to get a table out of this project is
still a Python prompt.

---

## The decisions this task has to make

**1. What can be exported?** Two scopes, and they are named, not implied.

*This run* and *every run in the project*. ADR-0048 built the second on purpose — *"statistics across
a dataset is why the measurements exist"* — and a menu item that silently means one of them is a
menu item somebody uses wrong once.

**2. Where does the file go?** Into the project's `exports/`, with the name the use case chooses.

Not a file dialog. The export is **part of the project** (ADR-0003's layout), it is timestamped so
today's does not replace yesterday's (ADR-0048), and asking an operator where to put a file they
have not seen yet is asking them to invent a filing system per export. What the window then does is
say **where it went**.

**3. What happens when there is nothing to export?** The refusal the use case already writes.

A detect-only run measures nothing, and `export_measurements` raises rather than writing headers with
no rows — because a file with headers and no rows says *"we measured and found nothing"*, which is a
different statement (ADR-0048). The window shows that sentence; it does not pre-empt it with a
disabled button that says less.

**4. Does it run as a job?** Yes. Reading every stored table in a project is disk, and the runner
has been there since M5-T07.

---

## Scope

**In scope**

1. `gui/viewmodels/session.py` — `export(scope)` as a job, and where the file went
2. `MainWindow` — *File → Export Measurements…* with the two scopes
3. **ADR-0067** — two named scopes, the project's own `exports/`, and a refusal that is a sentence
4. Tests: each scope writes what it names, the file lands in `exports/`, a detect-only run is refused
   with the use case's own message, and the window says where the file went

**Out of scope**

- **Choosing a directory outside the project** — decision 2; an export that leaves the project is a
  copy, and copying is what a file manager is for
- **Choosing columns** — the export's shape is ADR-0048's, provenance first
- **Exporting a figure** — `infrastructure.imaging.plots`, and no task has asked for it

---

## Definition of done

- [x] Both scopes export, as a job, into the project's `exports/`
- [x] The window says the path, relative to the project
- [x] Nothing to export is the use case's own sentence, not a silent no-op
- [x] ADR-0067 + the ADR index
- [x] `make check` green — 1135 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M6-T07: the export an operator can ask for, and the scope they mean`

---

## What it turned up

**Two test files with the same basename broke collection — for the second time.** `tests/gui/test_export.py`
against `tests/integration/test_export.py`, exactly the trap M5-T07 hit with `test_jobs.py`. Every
targeted run passed and `make check` could not collect. Renamed to `test_export_ui.py`, **and this
time the lesson ships as a guard**: `test_no_two_test_modules_share_a_basename` walks `tests/` and
fails on a clash. A lesson that lives only in a `Progress.md` entry is one the project gets to learn
twice.
