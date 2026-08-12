# ADR-0043 — Cancellation is asked for, not forced

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T06)
- **Affects:** `application/jobs`, `infrastructure/storage` · M4 · M5's GUI

## Context

Everything the application layer can do returns when it is finished. Architecture §4.5 set the bar
before any of it existed — *anything over ~100 ms runs as a job with progress and cancellation, and
the GUI subscribes rather than owning the thread policy* — and M5's exit criteria require a long
job that shows progress and can be cancelled without freezing the UI.

Building that in M5 would put the thread policy inside the widget that must not have it. Building
it here means deciding, in writing, what a cancel button is allowed to promise.

## Decision

### 1. A job is a handle, not a base class

`JobRunner.submit(name, work, listener)` returns a `Job`. `work` is any callable taking a
`JobContext`.

`DetectionJob(Job)` and `ImportJob(Job)` would differ only in which function they call, and a
hierarchy whose subclasses differ by a function is a function with extra steps. Third application
of ADR-0041's rule, and the one where the pull is strongest, because job hierarchies are what every
framework ships.

### 2. Threads, on `ThreadPoolExecutor`

The work is NumPy, SciPy and torch, all of which release the GIL where the time goes. A process
pool would have to pickle height maps across a boundary — expensive — and could not pass a loaded
SAM2 predictor across it at all.

The stdlib already gives submission, results, exceptions and cancelling a job that has not started.
What it does not give is progress and stopping one that has; those two are the whole of this
module.

### 3. Cancellation is cooperative, and there is no version of this that is not

A running Python thread cannot be killed. `Future.cancel()` works only while a job is still queued.
So:

- a **queued** job is dropped and goes straight to `CANCELLED`, having never run;
- a **running** job stops at its next `JobContext.raise_if_cancelled()`;
- a running job **with no checkpoint runs to completion**, and the request is recorded but never
  acted on.

That last row is the decision. A single twenty-second LoG pass has nowhere to check, so the cancel
button means *stop at the next checkpoint* — and the GUI must say that, because the alternative is
a button that appears to do nothing and an operator who concludes the application is frozen. Two
tests pin the two halves, deliberately next to each other.

### 4. Progress is counts, and `total = 0` means "cannot say"

`report(done, total, message)`. A batch knows "12 of 40" and a progress bar can divide; a single
opaque scientific call knows it is running and nothing more, and reporting an invented percentage
for it would be a number that means nothing moving at a rate that means less.

A successful job's progress is forced to `done == total` on completion: a bar left at 39 of 40
after the work is finished is a bar saying the work is not.

### 5. One listener, called on the worker thread

`submit(..., listener=...)` is called on every state and progress change — not a subscription
system with topics and unsubscribe, which no caller has asked for.

**It fires on the worker thread.** Qt widgets may be touched only from the main thread, so M5's
adapter marshals; forgetting is a crash rather than a warning. Stated in the module docstring, in
`submit`, and here, because it is the kind of fact that is obvious to whoever wrote it and invisible
to whoever uses it.

### 6. A failure is captured on the job

Every job body runs inside one wrapper that catches `BaseException`, stores it as `job.error`, and
sets `FAILED`. Nothing escapes into a `Future` nobody reads — the default way a thread pool loses a
traceback — and a job that failed silently is indistinguishable from one still running.

The exception object is kept, not a rendering of it: how much of it an operator should see is the
caller's decision.

### 7. The repository is usable from a worker thread, which it was not

`sqlite3.connect` defaults to `check_same_thread=True`: a connection refuses to be used from any
thread but the one that made it. A project opened on the main thread was therefore unusable inside
**every job** — found by the first test that ran an import under the runner, and otherwise destined
to arrive in M5 as a crash in a background task.

Two changes, both needed:

- `connect(..., check_same_thread=False)`. SQLite's own library is compiled *serialized* in
  CPython's build (`sqlite3.threadsafety == 3`), so sharing a connection is safe at the statement
  level.
- **One reentrant lock around every repository method.** Statement-level safety is not enough:
  `save_analysis` writes a run, its detections and a path in three statements, and a second thread
  committing between them would commit half of it. One lock for the whole repository, not one per
  table — this is a single-user desktop application, and the day contention is measurable the
  answer is a connection per thread, not a finer lock.

### 8. `import_images` is the first caller

It already loops over files, so it is the one place today where "12 of 40" exists and where
stopping is clean — **between** files. A half-copied scan with no row is exactly the litter
`check_integrity` reports.

A cancelled import keeps what it already imported. Those are real files with real rows, and the
report says what was done before the stop rather than pretending nothing happened.

## Consequences

**Positive**

- M5 can show progress and offer cancellation without owning a thread.
- A background failure is visible on the job instead of vanishing.
- The repository is thread-safe by construction rather than by convention, and a test says so.
- The honest limit of cancellation is written down where the GUI author will read it.

**Negative**

- Cancelling a long scientific pass does nothing until it ends. The fix would be checkpoints inside
  `core.science`, which means a progress callback in the domain layer — a bigger change than this
  task, and one that trades the purity of `core` for a UI affordance. Not taken; named here so the
  next person does not think it was missed.
- One lock serialises the repository, so two jobs writing to one project take turns. Irrelevant at
  one desktop user; measurable only in a scenario this application does not have.
- `listener` on the worker thread is a trap for exactly one caller — the GUI — and the mitigation
  is documentation plus M5's adapter.

**Neutral**

- `JobCancelled` is `concurrent.futures.CancelledError` rather than a new class: the stdlib already
  raises it when a queued job is dropped, so one `except` covers both ways a job can be cancelled.
- Jobs are live objects with no history. Persisting them is M4-T14's logging, not this.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A `Job` base class per operation | Subclasses differing by a function; ADR-0041's rule, third application |
| Processes instead of threads | Pickling height maps, and a loaded predictor cannot cross the boundary at all |
| Killing a running thread (`ctypes`, `PyThreadState_SetAsyncExc`) | Unsafe by design: leaves locks held and files half-written, and the CPython docs say so |
| An event bus with topics and unsubscribe | No caller has two listeners; a callback is what a callback is for |
| Progress as a float 0…1 | Loses "12 of 40", which is what the batch actually knows, and invites a fake percentage for an opaque step |
| `asyncio` | The work is blocking and CPU-bound in C; an event loop would need a thread pool underneath it anyway, plus a colour on every function |
| A connection per thread | Correct, and more machinery than one lock buys back at one user — the stated upgrade path if it ever matters |

## Compliance

- `tests/unit/test_jobs.py` covers success, failure captured, listener notification, cancelling a
  queued job, cancelling a running one at its checkpoint, and **a running job with no checkpoint
  finishing anyway** — the two halves of §3 side by side.
- Nothing in those tests sleeps to synchronise: `threading.Event` and `Barrier` make the meetings
  deterministic, so a slow machine cannot make them flake.
- `tests/integration/test_project_lifecycle.py` runs a batch import as a job, asserts the counts it
  reported, cancels it between files, and asserts the project is consistent afterwards.
- One test opens a project on the main thread and uses it inside a job — the case §7 fixes.

## References

- `docs/Architecture.md` §4.5 — the rule this implements
- ADR-0041 (a use case earns its place) — the rule §1 applies again
- ADR-0040 / ADR-0042 — the repository whose thread-safety §7 had to fix
- `docs/Roadmap.md` M5 — "a long-running job shows progress and can be cancelled"
