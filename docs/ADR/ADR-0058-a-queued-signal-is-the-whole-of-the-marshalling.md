# ADR-0058 — A queued signal is the whole of the marshalling

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M5-T07)
- **Affects:** `gui/viewmodels`, `gui/panels`, `gui/dialogs` · M5 · M6's analysis runs

## Context

ADR-0043 built the job runner and left one obligation, stated three times — in the module docstring,
in `submit`, and in the ADR:

> *`listener` … is called on every state and progress change, **on the worker thread**. Qt widgets
> may be touched only from the main thread, so M5's adapter marshals; forgetting is a crash rather
> than a warning.*

M5's third exit criterion is *"a long-running job shows progress and can be cancelled **without
freezing the UI**"*, and that clause rules out the implementation everybody writes first.

There was also nothing long to run: `import_images` is the one operation that already loops, already
counts "12 of 40", and already stops cleanly between files (ADR-0043 §8) — and no menu item could
call it. A progress bar demonstrated on a sleep loop is a progress bar for a thing nobody does.

## Decision

### 1. The marshalling is `listener=self.job_changed.emit`, and nothing else

Qt queues a signal whose receiver lives on another thread. The listener therefore does exactly one
thing — emit — and the thread crossing is the *connection*, not code we maintain.

No `QThread` subclass, no `moveToThread`, no worker object. The thread policy lives in
`application/jobs.py`; a second one in `gui/` would be two answers to one question, and the second
would be the one that drifts.

### 2. Progress lives in the status bar, not in a modal dialog

A modal progress dialog **is** the frozen window the criterion forbids: it blocks the window it is
reporting on. The strip — name, message, counts, Cancel — leaves the operator able to look at their
data while an import runs.

The bar shows **`%v of %m`**, the counts ADR-0043 §4 insisted on, not a percentage; `total == 0`
puts it into Qt's busy mode, because a determinate bar parked at 0 % that never moves is a lie about
the same fact.

### 3. The signal carries the handle, and the handle is read on delivery

`job_changed(Job)`, not `job_changed(Progress)`. A queued signal is delivered when the receiving
thread next looks at its event queue, so a widget reading `job.progress` in the slot sees the job as
it is **now**, not as it was when the update was emitted. A backlog therefore collapses to the
latest state — which is exactly what a progress bar wants, and the reason nothing here emits an
immutable snapshot that would repaint the bar with history.

**The same fact makes "on finish" handlers fire once per update ever emitted.** Every queued
delivery arrives after the job has finished and reports a finished job, so the session records the
job whose ending it has already dealt with and ignores the rest. This is written down because it is
invisible until something is done twice, and the something here was *refresh the project and
announce the outcome*.

### 4. Cancel asks, and the button says so

ADR-0043 §3 stands: a queued job is dropped, a running one stops at its next checkpoint, and one
with no checkpoint runs to completion. Once pressed, the button reads **"Stopping…"** and is
disabled.

That wording is the decision. A button that says "Cancelled" is one an operator presses, watches do
nothing for twenty seconds, and concludes the application has frozen — the exact impression the job
abstraction exists to prevent.

### 5. A cancelled import is a job that *succeeded*

`import_images` stops by **returning** its partial report, so the job's state machine reports
`SUCCEEDED` and only `cancellation_requested` records that somebody pressed the button. The summary
reads the request, not the state.

Left as it is rather than "fixed": the state describes the *job*, which did complete and did produce
a result; the request describes the *operator*. Making the use case raise `JobCancelled` instead
would throw the partial report away, and the files it copied are already on disk with rows pointing
at them.

### 6. One job at a time in this window

The runner accepts more (`max_workers=2`); this GUI submits one and refuses a second while one runs.
A status bar has one strip, and two jobs mean either two strips or one that silently describes the
newer.

While a job runs, **Open**, **Close** and **Import** are disabled: `close_project()` closes the
SQLite connection the worker thread is using. Cancel stays live.

### 7. The import asks two questions and invents neither

Modality, because nothing in a filename says it, and the pixel scale — whose minimum reads
**"unknown"** rather than `0.00` and comes back as `None`.

This is the first surface in the application that *creates* rows rather than reading them, so it is
where a fabricated scale would enter and never be noticed again (ADR-0025).

## Consequences

**Positive**

- ADR-0043's marshalling obligation is discharged in one expression, and a test asserts the thread
  the update arrives on rather than trusting the mechanism.
- The window stays usable during an import; the operator can pan the scan they already have open.
- The cancel button's promise matches what cancellation can actually do.
- M6 submits an analysis through the same three lines, and gets the strip for free.

**Negative**

- The "handle, not snapshot" choice means a *history* of progress cannot be reconstructed from the
  signal. Nothing wants one; if a log panel ever does (M5-T08), it needs its own record.
- One job at a time is a GUI limit, not a runner limit — the two disagree on purpose, and the day a
  second concurrent job is wanted the strip has to become a list.
- The window disables three actions while a job runs. That is coarser than it has to be: a job that
  never touches the repository would be blocked by the same rule.

**Neutral**

- `_settled` is one attribute doing the work of an idempotency guard. A stream of updates that all
  say the same terminal thing is what a queued connection produces, and this is the smallest honest
  answer to it.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A modal `QProgressDialog` | It is the frozen UI the exit criterion forbids |
| `QThread` / `moveToThread` in `gui/` | A second thread policy beside `application/jobs.py`, and the second one drifts |
| Emit an immutable `Progress` snapshot | Repaints the bar with history; the latest state is what a bar wants |
| Poll the job on a `QTimer` | A timer that fires when nothing happened, and lag on the update that matters |
| Make cancellation raise, so a cancelled import is `CANCELLED` | Throws away the partial report describing files already on disk |
| A demo "long job" instead of the import | A progress bar for a thing nobody does |
| Ask for the scale later, in a properties editor | The row is created here; a scale added later is a scale that was wrong in between |

## Compliance

- `tests/gui/test_background_jobs.py::TestTheListenerCrossesThreads` asserts every update arrives on
  `QApplication.instance().thread()`, and that the work itself did **not** run on it.
- The same file pins §3 (a backlog reads as the latest state), §4 (the button says *Stopping…*), §5
  (a cancelled import is `SUCCEEDED`, and the summary says "cancelled"), §6 and §7.
- `tests/gui/test_main_window.py::TestWhileAJobRuns` asserts Open, Close and Import go dead while a
  job runs and come back afterwards.
- No test sleeps to synchronise: a `threading.Event` inside the repository call makes the meeting
  deterministic, which is ADR-0043's own test discipline.

## References

- ADR-0043 (cancellation is asked for, not forced) — the obligation this discharges, §3, §4, §6, §8
- ADR-0057 §7 — the `QObject` this needed, created one task earlier for this reason
- ADR-0025 — why the scale field has an "unknown" that is a value
- `docs/Roadmap.md` M5 — *"a long-running job shows progress and can be cancelled"*
