# CURRENT TASK

**ID:** `M5-T07`
**Title:** A job that reports from another thread, and a cancel button that says what it means
**Milestone:** M5 — GUI shell, seventh task
**Defect:** — · **ADR:** **ADR-0058**
**Branch:** `feat/m5-gui-shell`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M5's third exit criterion: *"a long-running job shows progress and can be cancelled without freezing
the UI"*. It is the last piece of the shell that is not there, and it is the one M4 left an explicit
obligation for — ADR-0043 states **three times**, in the module docstring, in `submit`, and in the
ADR itself, that a job's listener fires **on the worker thread**:

> *Qt widgets may be touched only from the main thread, so M5's adapter marshals; forgetting is a
> crash rather than a warning.*

M5-T06 built the object that marshalling needs (ADR-0057 §7). This task uses it.

**There is no long-running job reachable from this GUI today.** `import_images` is the one operation
that already loops, already reports "12 of 40", and already stops cleanly between files (ADR-0043
§8) — and nothing in the window can call it. Demonstrating the criterion with a sleep loop would be
a progress bar for a thing nobody does, so the import action ships with the runner that makes it
long.

---

## The decisions this task has to make

**1. How does a worker-thread callback become a main-thread update?** A signal, and nothing else.

`runner.submit(name, work, listener=self.job_changed.emit)`. Qt queues a signal whose receiver lives
on another thread; the listener therefore does exactly one thing, and the marshalling is the
connection rather than code. No `QThread` subclass, no `moveToThread`, no worker object — the thread
policy lives in `application/jobs.py` and this layer does not get a second one.

**2. Where does progress appear?** In the status bar, not in a modal dialog.

A modal progress dialog *is* the frozen UI the criterion forbids — it blocks the window it is
reporting about. A status-bar strip with a label, a bar and a Cancel button leaves the operator able
to look at their data while the import runs, which is the whole point of a background job.

**3. What does the cancel button promise?** *Stop at the next checkpoint* — and it says so.

ADR-0043 §3: a queued job is dropped, a running one stops at its next `raise_if_cancelled()`, and
one with no checkpoint runs to completion. The button that hides that is the button an operator
presses twice and then concludes the application is frozen. After it is pressed it says
**"stopping…"** and stays disabled, because the honest report is *asked, not done*.

**4. Indeterminate is a state, not a zero.** `total == 0` means *cannot say* (ADR-0043 §4), and the
bar goes into Qt's busy mode rather than sitting at 0 % — a bar at 0 % that never moves is a lie
about the same fact.

**5. One job at a time in this window.** The runner takes more (`max_workers=2`); this GUI shows
one, and a second submission is refused while one is running.

A status bar has one strip. Two jobs mean either two strips or one that silently describes the
newer — and while a job is running the actions that would pull the project out from under it
(Open, Close, Import, Remove) are disabled, because `close_project()` closes the SQLite connection
the worker thread is using.

**6. The import needs to be asked two things**, and neither may be invented: the modality, and the
pixel scale. The scale field's zero reads **"unknown"** (Qt's `setSpecialValueText`), so absent is a
value the operator can choose rather than a blank they have to trust — ADR-0025 again, at the
surface that *creates* rows rather than reads them.

---

## Scope

**In scope**

1. `gui/viewmodels/session.py` — `job_changed`, `reported`, `import_images`, `cancel_job`, and the
   completion handling (refresh the project, say what the report said)
2. `gui/panels/job_status.py` — the status-bar strip: name, message, bar, Cancel
3. `gui/dialogs/import_images.py` — modality and scale, with "unknown" as a choosable value
4. `MainWindow` — File → *Import Images…*, the strip in the status bar, actions disabled while a
   job runs
5. **ADR-0058** — marshalling by signal, progress in the status bar, what cancel promises
6. Tests: the listener arriving on the **main thread**, determinate and indeterminate progress,
   cancellation between files with the copied files kept, a failed job as a message, one job at a
   time, and the explorer refreshing when an import finishes

**Out of scope**

- **A job history / log panel** — M5-T08 owns it; jobs are live objects with no history (ADR-0043)
- **Progress inside a single scientific pass** — ADR-0043's named negative consequence: it needs
  checkpoints in `core.science`, which is a callback in the domain layer
- **Running an analysis** — M6

---

## Definition of done

- [x] A worker-thread listener reaches a widget on the main thread, proven by a test that asserts
      the thread
- [x] Import runs as a job, with progress, and is cancellable; what was copied before the stop stays
- [x] The cancel button says *stop at the next checkpoint* rather than implying more
- [x] ADR-0058 + the ADR index
- [x] `make check` green — 998 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `Roadmap.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M5-T07: a job that reports from another thread, and a cancel button that says what it means`

---

## What it turned up

**A queued signal carries the handle, not a snapshot — so every "on finish" handler fires once per
update the job ever emitted.** The deliveries all arrive after the work is over and every one of
them reads a *finished* job: an import refreshed the project and announced its outcome five times.
Recording the job whose ending has been dealt with is the fix, and reading the handle late is the
right behaviour for the bar itself — a backlog collapses to the latest state, which is what a
progress bar wants. Found by a test that asserted a progress value and got the final one instead.

**A cancelled import is a job that SUCCEEDED.** `import_images` stops by *returning* its partial
report, so the state machine never sees a cancellation and `cancellation_requested` is the only
record that somebody pressed the button. The summary reads the request. Left as it is rather than
"fixed": the state describes the job, which did complete and did produce a result.

**Two test files with the same basename collide at collection.** `tests/unit/test_jobs.py` and a new
`tests/gui/test_jobs.py` — every targeted run passed and `make check` failed to *collect*. Renamed
to `test_background_jobs.py`; the same class of ambiguity M5-T02 hit with two `conftest.py`.

**A progress bar with its text off is an empty box.** Seen in the window, not in a test: at 0 of 6 it
renders as a blank rectangle. `%v of %m` shows the counts ADR-0043 §4 chose over a percentage.
