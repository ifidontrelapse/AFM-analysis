"""What a job promises, and the one thing it cannot (M4-T06, ADR-0043).

Concurrency tests earn their reputation by waiting on wall-clock time and then
failing on a loaded CI machine. Nothing here sleeps to synchronise: every test
that needs two threads to meet uses a `threading.Event`, so the test is as fast
as the machine and cannot be flaky in the direction that matters. Timeouts exist
only so a broken implementation fails in a second instead of hanging the suite.

The test that matters most is `test_a_running_job_stops_at_its_checkpoint`,
together with `test_a_job_with_no_checkpoint_runs_to_completion` — the two
halves of the truth about cancelling a Python thread.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from nanoscope.application.jobs import Job, JobCancelled, JobContext, JobRunner, JobState, Progress

#: Long enough that a busy machine still passes, short enough that a hang is a
#: failed test rather than a hung suite. Nothing waits for this in the happy path.
TIMEOUT = 5.0


@pytest.fixture
def runner() -> Iterator[JobRunner]:
    with JobRunner(max_workers=2) as jobs:
        yield jobs


class TestAJobThatFinishes:
    def test_it_returns_what_the_work_returned(self, runner: JobRunner) -> None:
        job = runner.submit("adding", lambda ctx: 2 + 2)

        assert job.wait(TIMEOUT)
        assert job.state is JobState.SUCCEEDED
        assert job.result == 4
        assert job.error is None

    def test_the_progress_ends_where_it_said_it_would(self, runner: JobRunner) -> None:
        """A bar left at 39 of 40 after the work is done is a bar that says the
        job is still going."""
        job = runner.submit("counting", lambda ctx: ctx.report(39, 40))

        assert job.wait(TIMEOUT)
        assert job.progress.done == 40
        assert job.progress.fraction == 1.0

    def test_progress_is_reported_as_it_happens(self, runner: JobRunner) -> None:
        seen: list[Progress] = []

        def work(ctx: JobContext) -> None:
            for index in range(3):
                ctx.report(index, 3, f"step {index}")

        job = runner.submit("stepping", work, listener=lambda j: seen.append(j.progress))
        assert job.wait(TIMEOUT)

        assert [(p.done, p.total) for p in seen if p.total] == [(0, 3), (1, 3), (2, 3), (3, 3)]

    def test_an_indeterminate_job_says_so_rather_than_guessing(self, runner: JobRunner) -> None:
        """One opaque scientific call knows it is running and nothing more."""
        job = runner.submit("thinking", lambda ctx: ctx.report(0, 0, "detecting"))

        assert job.wait(TIMEOUT)
        assert not Progress(0, 0).is_determinate
        assert Progress(0, 0).fraction is None


class TestAJobThatFails:
    def test_the_exception_is_kept_on_the_job(self, runner: JobRunner) -> None:
        """Not left unread in a `Future`, which is how a thread pool loses a
        traceback — and a job that failed silently is indistinguishable from one
        still running."""

        def work(ctx: JobContext) -> None:
            raise ValueError("the file was not a file")

        job = runner.submit("failing", work)

        assert job.wait(TIMEOUT)
        assert job.state is JobState.FAILED
        assert isinstance(job.error, ValueError)
        assert "not a file" in str(job.error)

    def test_a_failure_does_not_take_the_runner_with_it(self, runner: JobRunner) -> None:
        failing = runner.submit("failing", lambda ctx: 1 / 0)
        assert failing.wait(TIMEOUT)

        after = runner.submit("fine", lambda ctx: "still here")

        assert after.wait(TIMEOUT)
        assert after.result == "still here"

    def test_the_listener_hears_about_it(self, runner: JobRunner) -> None:
        states: list[JobState] = []

        job = runner.submit("failing", lambda ctx: 1 / 0, listener=lambda j: states.append(j.state))

        assert job.wait(TIMEOUT)
        assert states[-1] is JobState.FAILED


class TestCancelling:
    def test_a_running_job_stops_at_its_checkpoint(self, runner: JobRunner) -> None:
        """Cooperative cancellation, which is the only kind there is: the job
        stops where it asked whether it should."""
        started, done = threading.Event(), threading.Event()

        def work(ctx: JobContext) -> str:
            started.set()
            done.wait(TIMEOUT)  # let the test cancel us, then check
            ctx.raise_if_cancelled()
            return "should not get here"

        job = runner.submit("interruptible", work)
        assert started.wait(TIMEOUT)
        job.cancel()
        done.set()

        assert job.wait(TIMEOUT)
        assert job.state is JobState.CANCELLED
        assert job.result is None

    def test_a_job_with_no_checkpoint_runs_to_completion(self, runner: JobRunner) -> None:
        """The other half of the truth, and the reason the button says *ask*: a
        running Python thread cannot be killed. A twenty-second LoG pass that
        never checks will finish, and the request is remembered but never acted
        on."""
        started, release = threading.Event(), threading.Event()

        def work(ctx: JobContext) -> str:
            started.set()
            release.wait(TIMEOUT)  # stand in for a long pass with nothing to ask
            return "finished anyway"

        job = runner.submit("uninterruptible", work)
        assert started.wait(TIMEOUT)
        job.cancel()
        release.set()

        assert job.wait(TIMEOUT)
        assert job.cancellation_requested
        assert job.state is JobState.SUCCEEDED
        assert job.result == "finished anyway"

    def test_a_job_that_never_started_is_dropped(self) -> None:
        """One worker, one job holding it, so the second is still queued — the
        case where the stdlib can cancel outright and nothing runs at all."""
        blocked, release = threading.Event(), threading.Event()

        with JobRunner(max_workers=1) as runner:
            first = runner.submit("blocking", lambda ctx: (blocked.set(), release.wait(TIMEOUT)))
            assert blocked.wait(TIMEOUT)
            ran = threading.Event()
            queued = runner.submit("queued", lambda ctx: ran.set())

            queued.cancel()
            release.set()

            assert first.wait(TIMEOUT)
            assert queued.state is JobState.CANCELLED
            assert not ran.is_set()

    def test_cancelling_a_finished_job_changes_nothing(self, runner: JobRunner) -> None:
        job = runner.submit("quick", lambda ctx: "done")
        assert job.wait(TIMEOUT)

        job.cancel()

        assert job.state is JobState.SUCCEEDED
        assert job.result == "done"

    def test_the_checkpoint_is_the_job_asking(self) -> None:
        """`raise_if_cancelled` on its own, without a runner: it raises only
        after somebody asks, and what it raises is the stdlib's cancellation."""
        job = Job("standalone")
        context = JobContext(job)

        context.raise_if_cancelled()
        job.cancel()

        with pytest.raises(JobCancelled):
            context.raise_if_cancelled()


class TestTheRunner:
    def test_two_jobs_run_at_once(self) -> None:
        """Two workers, two jobs that only finish once both have started —
        deadlocks if the pool is serialising them."""
        both = threading.Barrier(3, timeout=TIMEOUT)

        with JobRunner(max_workers=2) as runner:
            first = runner.submit("a", lambda ctx: both.wait())
            second = runner.submit("b", lambda ctx: both.wait())
            both.wait()

            assert first.wait(TIMEOUT)
            assert second.wait(TIMEOUT)
            assert (first.state, second.state) == (JobState.SUCCEEDED, JobState.SUCCEEDED)

    def test_every_job_has_its_own_identity(self, runner: JobRunner) -> None:
        first = runner.submit("same name", lambda ctx: None)
        second = runner.submit("same name", lambda ctx: None)

        assert first.id != second.id

    def test_shutdown_waits_for_what_is_running(self) -> None:
        finished = threading.Event()

        runner = JobRunner(max_workers=1)
        job = runner.submit("slow", lambda ctx: finished.set())
        runner.shutdown()

        assert finished.is_set()
        assert job.is_finished
