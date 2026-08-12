# CURRENT TASK

**ID:** `M4-T06`
**Title:** Jobs: submit, progress, cancel, and the cancellation that cannot be forced
**Milestone:** M4 — Application layer, sixth task
**Defect:** — (W1: no application layer exists) · **ADR:** **ADR-0043**
**Branch:** `feat/m4-application-layer` — M4 changes no scientific output (PROJECT_RULES §7)
**Status:** **done 2026-08-12.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

Everything the application layer can do so far returns when it is finished. Importing forty scans
copies forty files; `run_analysis` runs a full LoG pass and a measurement sweep. Architecture §4.5
set the bar before any of it existed:

> *Anything that can take longer than ~100 ms … runs as a job with progress and cancellation. The
> GUI subscribes; it does not block and it does not own the thread policy.*

M5's own exit criteria require a long job that shows progress and cancels without freezing the UI.
Building that in M5 would put the thread policy inside the widget that must not own it.

---

## The decisions this task has to make

**1. What is a job, structurally?** A **handle**, returned by a runner — not a base class to
inherit. `DetectionJob(Job)` and `ImportJob(Job)` would be two classes whose only difference is
which function they call, which is ADR-0041's rule for the third time. A job wraps *any* callable.

**2. Threads or processes?** Threads, on `concurrent.futures.ThreadPoolExecutor`.

The work is NumPy, SciPy and torch, all of which release the GIL in the parts that take the time.
A process pool would have to pickle height maps and model handles across a boundary — expensive
for the arrays and impossible for a loaded SAM2 predictor. Stdlib gives submit, result, exception
and pre-start cancellation for free; what it does not give is the next two rows.

**3. How does cancellation work?** **Cooperatively, and this is the honest part of the task.**

A running Python thread cannot be killed. `Future.cancel()` works only while a job is still
queued. So a job that has started stops at the next point where it *checks* — the runner hands
every job a `JobContext` with `raise_if_cancelled()`, and the job calls it where stopping is safe.
A single 20-second LoG pass has no such point, and pretending otherwise would be the lie the GUI
then repeats to the operator: the cancel button must mean "stop at the next checkpoint", not
"stop".

**4. Where does progress come from?** The same `JobContext`: `report(done, total, message)`.
Counting, not a fraction — a batch of forty files knows "12 of 40", and turning that into a
fraction is the progress bar's business. `total=0` means indeterminate, which is what a single
opaque scientific call actually is.

**5. Who is told, and on which thread?** One optional `listener` callback per job, called on every
state or progress change. **It fires on the worker thread**, which is stated loudly here and in
the ADR because Qt widgets may only be touched from the main thread — marshalling is M5's adapter
to write, and a GUI that forgets is a crash, not a warning.

**6. What does a failure do?** It is caught, stored on the job as its `error`, and the job goes to
`FAILED`. A job that dies must not take the runner with it, and the exception must not vanish into
a thread nobody joins — which is what `ThreadPoolExecutor` does by default if nobody reads the
future.

**7. Does anything actually use it?** `import_images` gains an optional `progress` parameter: it
already loops over files, so it is the one place today where "12 of 40" exists to be reported, and
where a cancellation between files is clean. Files already imported stay imported, and the report
says what was done before the stop.

---

## Scope

**In scope**

1. `application/jobs.py` — `JobState`, `Progress`, `JobContext`, `Job`, `JobRunner`
2. `import_images(..., progress=None)` — progress and a cancellation checkpoint per file
3. **ADR-0043** — a handle not a hierarchy, threads, cooperative cancellation, the listener thread
4. Tests: success, failure captured, cancel before start, cooperative cancel while running,
   progress observed in order, the runner shutting down, and a cancelled batch import

**Out of scope**

- **Job persistence and history.** Logs are M4-T14; a job is a live object
- **Priorities, dependencies, retries, ETA.** No caller has asked, and each is a queue policy that
  wants a real workload to be designed against
- **Threading `JobContext` through `core.science`.** Progress inside a LoG pass means a callback
  in the domain, which `core` must not grow for the GUI's benefit. What that costs — one long
  opaque step — is stated rather than engineered around
- **Qt marshalling** — M5's adapter, named in decision 5

---

## Expected blast radius

- **Zero golden differences.** No numerical code is touched; `import_images` gains an optional
  parameter and keeps its behaviour when it is absent
- One new application module, one ADR, one test file
- No new dependency — `concurrent.futures` and `threading` are stdlib

---

## Definition of done

- [x] `JobRunner` over `ThreadPoolExecutor`, with `Job` as a handle
- [x] Cooperative cancellation, and a test that proves a *running* job stops at its checkpoint
- [x] Progress reported as counts, with the listener documented as worker-thread
- [x] Failures captured on the job rather than lost in a thread
- [x] `import_images` reporting progress and stopping between files
- [x] ADR-0043
- [x] `make check` green — 619 tests, golden byte-identical
- [x] `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [x] Commit: `M4-T06: a job reports, and stops when it is asked`

---

## What it turned up

**The repository could not be used from a job at all.** Python's `sqlite3` binds a connection to
the thread that created it and refuses it everywhere else, so a project opened on the main thread
was unusable inside **every** background task — found by the first test that ran an import under
the runner, and otherwise destined to arrive in M5 as a crash inside a widget's worker.

Both halves were needed: `check_same_thread=False`, because SQLite's own library is compiled
serialized in CPython's build, **and** one reentrant lock around every repository method, because
statement-level safety is not enough when `save_analysis` writes three statements that a second
thread could commit half of.

**Two of this task's tests were races when first written**, both fixed by making the threads meet
on an `Event` rather than a `sleep`. It is now the rule for the file: nothing in `test_jobs.py`
synchronises on wall-clock time, so a loaded machine cannot make it flake, and the timeouts exist
only so a broken implementation fails in a second instead of hanging the suite.

**`JobCancelled` did not need to be a new class.** `concurrent.futures.CancelledError` is what the
stdlib already raises when a queued job is dropped, so one `except` covers both ways a job ends
early.

---

## Notes

The golden held for the sixth time. **M4-T07** takes annotations — the first thing an operator
creates rather than the application deriving it, and the trigger ADR-0041 named for revisiting
import deduplication.
