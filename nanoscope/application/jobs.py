"""Work that takes long enough to need an answer to "how far?" (M4-T06).

Architecture §4.5 set the rule before there was anything to run under it:
anything over ~100 ms is a job with progress and cancellation, and the GUI
subscribes rather than owning the thread policy. This is where that policy
lives.

It is deliberately thin. `concurrent.futures.ThreadPoolExecutor` already does
submission, results, exceptions and cancelling a job that has not started; what
it does not do is progress, and stopping one that has. Those two are the whole
of this module (ADR-0043).

**Cancellation is cooperative, and there is no version of this that is not.** A
running Python thread cannot be killed. A job that has started stops at the next
point where it *asks* — `JobContext.raise_if_cancelled()` — so "cancel" means
*stop at the next checkpoint*, and a single twenty-second LoG pass has no
checkpoint at all. That is stated here, in the ADR, and in the docstring of
every method that touches it, because it is the fact a progress dialog is most
tempted to misrepresent.

**A listener runs on the worker thread.** Qt widgets may be touched only from
the main one, so the GUI's adapter marshals (M5). Forgetting is a crash, not a
warning.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any


class JobState(StrEnum):
    """Where a job is. Terminal states are the last three."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL = (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


#: What a job raises on its way out when it notices it has been asked to stop.
#: The stdlib's, not one of ours: `concurrent.futures` already raises exactly
#: this when a queued job is dropped, so one `except CancelledError` covers both
#: ways a job can be cancelled. It is deliberately **not** a `NanoscopeError` —
#: nothing was wrong with the input and no analysis failed; somebody pressed
#: cancel.
JobCancelled = CancelledError


@dataclass(frozen=True)
class Progress:
    """How far along a job says it is.

    Counts rather than a fraction: a batch knows "12 of 40", and a progress bar
    can divide. `total = 0` means *indeterminate*, which is the honest report
    from a single opaque scientific call — it knows it is running and nothing
    more.
    """

    done: int = 0
    total: int = 0
    message: str = ""

    @property
    def is_determinate(self) -> bool:
        return self.total > 0

    @property
    def fraction(self) -> float | None:
        """`done / total`, or `None` when the job cannot say."""
        return self.done / self.total if self.total > 0 else None


class JobContext:
    """What a running job is handed: a way to report, and a way to be stopped.

    The job's own code decides where stopping is safe, which is why the check is
    a call and not something the runner can do behind its back.
    """

    def __init__(self, job: Job) -> None:
        self._job = job

    @property
    def cancelled(self) -> bool:
        """True once somebody has asked this job to stop."""
        return self._job.cancellation_requested

    def raise_if_cancelled(self) -> None:
        """Stop here if asked. Call it where the work can be abandoned safely.

        Raises:
            JobCancelled: the runner catches it and marks the job `CANCELLED`.
        """
        if self.cancelled:
            raise JobCancelled(f"job {self._job.name!r} was cancelled")

    def report(self, done: int, total: int = 0, message: str = "") -> None:
        """Say how far along the work is. Cheap, and safe to call in a loop."""
        self._job.set_progress(Progress(done=done, total=total, message=message))


class Job:
    """A handle on one piece of work: its state, its progress, and its result.

    Not a base class. `DetectionJob(Job)` and `ImportJob(Job)` would differ only
    in which function they call, and a hierarchy whose subclasses differ by a
    function is a function with extra steps (ADR-0043, and ADR-0041's rule for
    the third time). A job wraps any callable that accepts a `JobContext`.

    Every attribute may be read from another thread; the lock makes state and
    progress move together, so a listener never sees `SUCCEEDED` with the
    progress of the step before it.
    """

    def __init__(self, name: str, listener: Callable[[Job], None] | None = None) -> None:
        self.id = str(uuid.uuid4())
        self.name = name
        self._listener = listener
        self._lock = threading.Lock()
        self._state = JobState.PENDING
        self._progress = Progress()
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._result: Any = None
        self._error: BaseException | None = None
        self._future: Future[Any] | None = None

    # ── What a caller reads ───────────────────────────────────────────────────

    @property
    def state(self) -> JobState:
        with self._lock:
            return self._state

    @property
    def progress(self) -> Progress:
        with self._lock:
            return self._progress

    @property
    def result(self) -> Any:
        """What the job returned, or `None` if it did not finish successfully."""
        with self._lock:
            return self._result

    @property
    def error(self) -> BaseException | None:
        """Why it failed, or `None`. The exception itself, not a rendering of it —
        the caller decides how much of it an operator should see."""
        with self._lock:
            return self._error

    @property
    def is_finished(self) -> bool:
        return self.state in _TERMINAL

    @property
    def cancellation_requested(self) -> bool:
        return self._cancel.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the job reaches a terminal state. Returns whether it did.

        For tests and for headless callers. A GUI does not call this — that is
        the freeze the job abstraction exists to prevent.
        """
        return self._done.wait(timeout)

    # ── What a caller does ────────────────────────────────────────────────────

    def cancel(self) -> None:
        """Ask the job to stop. **Ask**, because it cannot be made to.

        A job that has not started is dropped and goes straight to `CANCELLED`.
        A job that is running stops at its next `raise_if_cancelled()` — and if
        it has none, it runs to completion and the request is remembered but
        never acted on. Cancelling a finished job does nothing.
        """
        if self.is_finished:
            return
        self._cancel.set()
        if self._future is not None and self._future.cancel():
            self._finish(JobState.CANCELLED)

    # ── What the runner does. Not part of the caller's surface. ───────────────

    def set_progress(self, progress: Progress) -> None:
        with self._lock:
            self._progress = progress
        self._notify()

    def _start(self) -> None:
        with self._lock:
            if self._state is not JobState.PENDING:
                return
            self._state = JobState.RUNNING
        self._notify()

    def _finish(
        self,
        state: JobState,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        with self._lock:
            if self._state in _TERMINAL:
                return
            self._state = state
            self._result = result
            self._error = error
            if state is JobState.SUCCEEDED and self._progress.is_determinate:
                # A successful job is finished, whatever its last report said.
                # A bar left at 39/40 after the work is done is a bar that
                # says the job is still going.
                self._progress = replace(self._progress, done=self._progress.total)
        self._done.set()
        self._notify()

    def _notify(self) -> None:
        """Tell the listener something changed. **On the worker thread.**"""
        if self._listener is not None:
            self._listener(self)


class JobRunner:
    """Runs jobs on a thread pool, and hands back handles.

    Threads rather than processes: the work is NumPy, SciPy and torch, which
    release the GIL where the time goes, and a process pool would have to pickle
    height maps across a boundary and could not pass a loaded predictor across
    it at all (ADR-0043).
    """

    def __init__(self, max_workers: int = 2) -> None:
        #: Two by default: enough that a long analysis does not block an import,
        #: few enough that two scientific passes are not competing for the same
        #: cores. Not a policy anybody has measured — the number is here to be
        #: set by the composition root when there is something to measure.
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nanoscope")

    def submit(
        self,
        name: str,
        work: Callable[[JobContext], Any],
        listener: Callable[[Job], None] | None = None,
    ) -> Job:
        """Run `work` in the background and return the handle to it.

        Args:
            name: what to call this job where an operator will read it.
            work: a callable taking a `JobContext`. Report progress and check
                for cancellation through it.
            listener: called on every state and progress change, **on the worker
                thread**. A Qt caller marshals to the main thread (M5).

        Returns:
            A `Job`, possibly already running by the time this returns.
        """
        job = Job(name, listener)
        job._future = self._executor.submit(self._run, job, work)
        return job

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting jobs, and optionally wait for the running ones.

        `wait=False` returns immediately, but running jobs are *not* killed —
        they cannot be. It stops the queue, not the work.
        """
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> JobRunner:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    @staticmethod
    def _run(job: Job, work: Callable[[JobContext], Any]) -> Any:
        """The wrapper every job runs inside: one place that decides its fate.

        Nothing escapes. An exception here would otherwise sit unread in a
        `Future` nobody looks at, which is the default way a thread pool loses a
        traceback — and a job that failed silently is indistinguishable from one
        still running.
        """
        context = JobContext(job)
        try:
            context.raise_if_cancelled()
            job._start()
            result = work(context)
        except JobCancelled:
            job._finish(JobState.CANCELLED)
            return None
        except BaseException as exc:  # the point of this handler is that nothing escapes
            job._finish(JobState.FAILED, error=exc)
            return None
        job._finish(JobState.SUCCEEDED, result=result)
        return result
