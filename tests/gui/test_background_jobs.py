"""A job reporting from a worker thread into a window (M5-T07, ADR-0058).

M5's third exit criterion — *"a long-running job shows progress and can be
cancelled without freezing the UI"* — and the obligation ADR-0043 wrote down
three times: the listener fires on the **worker** thread, and Qt widgets may be
touched only from the main one.

So the first test is the one that matters: it records which thread the update
arrives on. Nothing here sleeps to synchronise — a `threading.Event` inside the
repository call makes the meeting deterministic, the way M4-T06's tests do it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.application.jobs import Job, JobState
from nanoscope.core.values import Modality
from nanoscope.gui.panels.job_status import STOPPING, JobStatus
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


@pytest.fixture
def scans(tmp_path: Path) -> list[Path]:
    """Three importable files, outside any project."""
    source = tmp_path / "source"
    source.mkdir()
    paths = []
    for name in ("monday.npy", "tuesday.npy", "wednesday.npy"):
        path = source / name
        np.save(path, np.zeros((8, 8), dtype=np.float32))
        paths.append(path)
    return paths


@pytest.fixture
def session(tmp_path: Path) -> Iterator[SessionViewModel]:
    root = tmp_path / "P"
    SqliteProjectRepository.create(root, "P").close()
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        model = SessionViewModel(container)
        model.open_project(root)
        yield model


def settle(job: Job) -> None:
    """Wait for the job, then let Qt deliver what it queued.

    The delivery is the point: a queued signal sits in the receiving thread's
    event queue until that thread looks, and in a test the looking is explicit.
    """
    assert job.wait(5.0)
    QApplication.processEvents()


class TestTheListenerCrossesThreads:
    def test_the_update_arrives_on_the_main_thread(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        """ADR-0043's obligation, discharged. Without the queued connection this
        assertion fails, and in a real window the failure is a crash in a
        background thread rather than a red test."""
        threads: list[QThread] = []
        session.job_changed.connect(lambda _job: threads.append(QThread.currentThread()))

        job = session.import_images(scans, modality=Modality.AFM)

        assert job is not None
        settle(job)
        assert threads
        assert set(threads) == {QApplication.instance().thread()}

    def test_the_work_itself_did_not_run_on_the_main_thread(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        """The other half: if the import ran inline, "without freezing the UI"
        would be met by a window that had nothing to do."""
        where: list[int] = []
        repository = session._app.repository
        assert repository is not None
        original = repository.import_image

        def watched(*args: object, **kwargs: object) -> object:
            where.append(threading.get_ident())
            return original(*args, **kwargs)  # type: ignore[arg-type]

        repository.import_image = watched  # type: ignore[method-assign]
        job = session.import_images(scans, modality=Modality.AFM)

        assert job is not None
        settle(job)
        assert where and threading.get_ident() not in where


class TestProgress:
    def test_it_counts_files_and_finishes_full(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        job = session.import_images(scans, modality=Modality.AFM)

        assert job is not None
        settle(job)
        assert job.progress.done == job.progress.total == 3

    def test_the_delivered_state_is_the_current_one_not_the_emitted_one(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        """A queued signal carries the **handle**, so a widget reads the job when
        the event is *delivered*. A backlog therefore collapses to the latest
        state — which is what a progress bar wants, and why nothing here emits a
        snapshot that would repaint the bar with history (ADR-0058 §3)."""
        seen: list[tuple[int, int]] = []
        session.job_changed.connect(
            lambda job: seen.append((job.progress.done, job.progress.total))
        )

        job = session.import_images(scans, modality=Modality.AFM)

        assert job is not None
        settle(job)
        assert seen and set(seen) == {(3, 3)}

    def test_the_strip_follows_a_running_job_and_hides_when_it_ends(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        """The bar is asserted **while it runs**: a job that finishes before the
        first delivery is one the strip never shows at all, and a strip that
        flashes for three milliseconds is noise rather than progress."""
        strip = JobStatus(session)
        assert not strip.isVisible()
        held = _held_import(session)

        job = session.import_images(scans, modality=Modality.AFM)
        assert job is not None
        assert held.in_flight.wait(5.0)
        QApplication.processEvents()

        assert strip.isVisible()
        assert strip.bar.maximum() == 3
        #: Counts rather than a percentage: "2 of 40" is what the job knows,
        #: and a percentage is a division the operator did not ask for.
        assert strip.bar.format() == "%v of %m"

        held.release.set()
        settle(job)
        assert strip.isHidden()

    def test_a_job_that_cannot_count_gets_a_busy_bar(self, session: SessionViewModel) -> None:
        """`total == 0` means *cannot say* (ADR-0043 §4). A determinate bar
        parked at 0 % that never moves is a lie about the same fact."""
        strip = JobStatus(session)
        job = session._app.jobs.submit(
            "thinking", lambda ctx: ctx.report(0, 0, "working"), listener=session.job_changed.emit
        )
        session._job = job
        settle(job)

        strip.show_job(_Reporting(job, done=0, total=0))

        assert (strip.bar.minimum(), strip.bar.maximum()) == (0, 0)


class TestCancelling:
    def test_it_stops_between_files_and_keeps_what_was_copied(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        """Between files is the only clean place to stop; the files that made it
        are real files with real rows (ADR-0043 §8)."""
        repository = session._app.repository
        assert repository is not None
        held = _held_import(session)

        job = session.import_images(scans, modality=Modality.AFM)
        assert job is not None
        assert held.in_flight.wait(5.0)

        session.cancel_job()
        held.release.set()
        settle(job)

        assert len(job.result.imported) == 1
        assert len(repository.list_images()) == 1
        assert repository.check_integrity().is_clean

    def test_a_cancelled_import_is_a_job_that_succeeded(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        """The finding this task turned up. `import_images` stops by *returning*
        its partial report, so the state machine says SUCCEEDED — the request is
        the only record that somebody pressed the button, and the summary reads
        it rather than the state."""
        said: list[str] = []
        session.reported.connect(said.append)
        job = session.import_images(scans, modality=Modality.AFM)
        assert job is not None
        job.cancel()
        settle(job)

        assert job.state is JobState.SUCCEEDED
        assert job.cancellation_requested
        assert "cancelled" in said[-1]

    def test_the_button_says_stopping_rather_than_stopped(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        """A queued job is dropped, a running one stops at its next checkpoint,
        and one with no checkpoint runs to the end — so the button that has been
        pressed reports a *request* (ADR-0043 §3)."""
        strip = JobStatus(session)
        job = session.import_images(scans, modality=Modality.AFM)
        assert job is not None

        strip._cancel_pressed()

        assert strip.cancel.text() == STOPPING
        assert not strip.cancel.isEnabled()
        assert job.cancellation_requested
        settle(job)


class TestOneJobAtATime:
    def test_a_second_submission_is_refused_while_one_runs(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        """A status bar has one strip; two jobs mean two, or one that silently
        describes the newer."""
        held = _held_import(session)

        first = session.import_images(scans, modality=Modality.AFM)
        assert first is not None
        assert held.in_flight.wait(5.0)

        assert session.import_images(scans, modality=Modality.AFM) is None
        assert session.is_busy

        held.release.set()
        settle(first)
        assert not session.is_busy

    def test_importing_without_a_project_does_nothing(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        session.close_project()

        assert session.import_images(scans, modality=Modality.AFM) is None


class TestWhatTheProjectSeesAfterwards:
    def test_the_panels_are_told_the_project_changed(
        self, session: SessionViewModel, scans: list[Path]
    ) -> None:
        announced: list[object] = []
        session.project_changed.connect(announced.append)

        job = session.import_images(scans, modality=Modality.AFM)

        assert job is not None
        settle(job)
        assert session.project is not None
        assert len(session.project.images) == 3
        assert announced

    def test_a_refused_file_is_counted_and_not_hidden(
        self, session: SessionViewModel, scans: list[Path], tmp_path: Path
    ) -> None:
        """`import_images` collects failures instead of aborting the batch
        (ADR-0041), and a batch that quietly imports 3 of 4 is one an operator
        finds out about much later."""
        said: list[str] = []
        session.reported.connect(said.append)

        job = session.import_images([*scans, tmp_path / "not-there.npy"], modality=Modality.AFM)

        assert job is not None
        settle(job)
        assert "Imported 3 file(s)" in said[-1]
        assert "1 refused" in said[-1]

    def test_a_job_that_raises_becomes_a_failure_message(self, session: SessionViewModel) -> None:
        """Nothing escapes into a `Future` nobody reads (ADR-0043 §6); what the
        window gets is a sentence."""
        said: list[str] = []
        session.failed.connect(said.append)

        def boom(_ctx: object) -> None:
            raise RuntimeError("the detector fell over")

        job = session._app.jobs.submit("boom", boom, listener=session.job_changed.emit)
        session._job = job
        settle(job)

        assert job.state is JobState.FAILED
        assert said == ["the detector fell over"]


class _HeldImport:
    """A repository whose first import blocks until the test lets it go.

    An `Event` rather than a sleep: the meeting happens at a known point, so a
    slow machine cannot make these tests flake (ADR-0043's own test discipline).
    """

    def __init__(self) -> None:
        self.in_flight = threading.Event()
        self.release = threading.Event()


def _held_import(session: SessionViewModel) -> _HeldImport:
    repository = session._app.repository
    assert repository is not None
    original = repository.import_image
    held = _HeldImport()

    def slow(*args: object, **kwargs: object) -> object:
        held.in_flight.set()
        held.release.wait(5.0)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    repository.import_image = slow  # type: ignore[method-assign]
    return held


class _Reporting:
    """A job whose progress is whatever the test says, for the busy-bar case."""

    def __init__(self, job: Job, *, done: int, total: int) -> None:
        self._job = job
        self.progress = type(job.progress)(done=done, total=total)
        self.name = job.name
        self.is_finished = False
        self.cancellation_requested = False


class TestTheQuestionsAnImportAsks:
    """Two things it cannot guess, and one of them may be absent (ADR-0025)."""

    def test_the_scale_can_be_answered_unknown(self) -> None:
        """`0` is not a scale, it is the fabricated one a milestone was spent
        removing — so the bottom of the range reads "unknown" and comes back as
        `None`. This is the first surface that *creates* rows, which is where an
        invention would start."""
        from nanoscope.gui.dialogs import ImportOptions

        dialog = ImportOptions()

        assert dialog.pixel_size.value() == 0.0
        assert dialog.choice().pixel_size_nm is None
        assert dialog.pixel_size.specialValueText() == "unknown"

    def test_a_stated_scale_comes_back_as_a_number(self) -> None:
        from nanoscope.gui.dialogs import ImportOptions

        dialog = ImportOptions()

        dialog.pixel_size.setValue(1.95)

        assert dialog.choice().pixel_size_nm == pytest.approx(1.95)

    def test_every_modality_can_be_chosen(self) -> None:
        from nanoscope.gui.dialogs import ImportOptions

        dialog = ImportOptions()

        offered = [dialog.modality.itemData(i) for i in range(dialog.modality.count())]

        assert offered == list(Modality)
        assert dialog.choice().modality in Modality


class TestWhatTheLogKeeps:
    def test_an_import_is_written_down_as_well_as_shown(
        self, session: SessionViewModel, scans: list[Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A status line lasts until the next one. ADR-0051 set the project's
        log aside to answer *what happened to this work*, and until M5-T08 put a
        panel on screen nothing noticed that an import wrote nothing into it."""
        with caplog.at_level(logging.INFO, logger="nanoscope.gui.viewmodels.session"):
            job = session.import_images(scans, modality=Modality.AFM)
            assert job is not None
            settle(job)

        assert any("Imported 3 file(s)" in record.message for record in caplog.records)
