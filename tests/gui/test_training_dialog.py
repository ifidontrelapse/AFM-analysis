"""Annotations become a model, from the window (M8-T05, ADR-0085).

Four tasks built a training module nothing called; these tests are the first
that press the button. The subject is the whole path — annotations → dataset →
run → a registered `ModelDescriptor` — driven through the viewmodel and the
dialog, with `FakeTrainingProvider` standing in for six hours of ultralytics.

**The fake is swapped into the container, not into the dialog.** `Nanoscope.open`
constructs a `LocalTrainingProvider`, and `app.training` is the seam a test
replaces — which is the same seam M8-T07's remote provider will arrive through.

Four things carry the task and none is about the layout:

- the whole path runs and **registers a model**, which is ADR-0006's compliance
  clause reaching a window;
- a run with nothing held out **shows no validation columns**, because ADR-0082
  says there is nothing honest to put in them;
- **`is_busy` and `is_training` are different questions** — an operator can
  annotate and undo while a model trains, and cannot close the project;
- a stored `running` run nobody is running reads as **interrupted**, never as
  failed (ADR-0084 §8).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from nanoscope.app.container import Nanoscope
from nanoscope.application.jobs import Job
from nanoscope.core.entities.model import ModelDescriptor, ModelTask
from nanoscope.core.entities.project import AnnotationSource
from nanoscope.core.entities.training import (
    DatasetSpec,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from nanoscope.core.values import Modality
from nanoscope.gui import main_window
from nanoscope.gui.dialogs.training import INTERRUPTED, TrainingDialog
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

#: The fake lives beside the contract it satisfies, and `tests/` is not a
#: package — the same two lines `test_dataset_builder.py` uses to reach it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
from fake_provider import FakeTrainingProvider

pytestmark = pytest.mark.usefixtures("qt_app")

#: The two answers the close question has. Named, because
#: `StandardButton.Close` in an assertion reads as a verb.
_CLOSE = QMessageBox.StandardButton.Close
_CANCEL = QMessageBox.StandardButton.Cancel

SCANS = 4


def phantom(seed: int, size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    z = 0.05 * x + 0.02 * y + rng.normal(0.0, 0.2, (size, size)).astype(np.float32)
    for _ in range(4):
        cy, cx = rng.integers(8, size - 8, 2)
        z[cy - 3 : cy + 3, cx - 3 : cx + 3] += 12.0
    return z.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for index in range(SCANS):
            source = tmp_path / f"scan{index}.npy"
            np.save(source, phantom(index))
            record = repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
            repo.add_annotation(
                record.id,
                label="particle",
                box=(4.0, 4.0, 16.0, 16.0),
                source=AnnotationSource.MANUAL,
            )
    return root


@pytest.fixture
def app(tmp_path: Path, project: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        container.open(project)
        #: The seam. `open` built a `LocalTrainingProvider`; a test that let it
        #: stand would be downloading weights and training a network.
        container.training = FakeTrainingProvider(Path(project))
        yield container


@pytest.fixture
def session(app: Nanoscope) -> SessionViewModel:
    return SessionViewModel(app)


def settle(job: Job | None) -> None:
    """Wait for the build job, then let Qt deliver what it queued."""
    assert job is not None
    assert job.wait(10.0)
    QApplication.processEvents()


def finish(session: SessionViewModel, job: Job | None) -> TrainingRun:
    """Wait for the whole thing: the build job, then the run behind it."""
    settle(job)
    for _ in range(2000):
        QApplication.processEvents()
        run = session.training
        if run is not None and run.is_finished:
            return run
        _pause()
    raise AssertionError("the run never reached a terminal state")


def _pause() -> None:
    """Let the fake's thread get an epoch further. Milliseconds, not a sleep
    loop that hides a hang: `finish` gives up after two thousand of these."""
    import time

    time.sleep(0.005)


def train(session: SessionViewModel, **overrides: object) -> TrainingRun:
    options: dict[str, object] = {
        "model_id": "particles-v1",
        "hand_drawn_only": True,
        "val_fraction": 0.25,
    }
    options.update(overrides)
    config = TrainingConfig(base_model="fake.pt", epochs=3, image_size_px=32)
    return finish(session, session.train(config, **options))  # type: ignore[arg-type]


class TestTheWholePathRunsFromHere:
    def test_it_produces_a_registered_model(
        self, session: SessionViewModel, app: Nanoscope
    ) -> None:
        """ADR-0006's compliance clause, reaching a window: annotations →
        dataset → weights → a `ModelDescriptor`, without leaving the
        application. M8's first exit criterion is this assertion."""
        run = train(session)

        assert run.status is TrainingStatus.SUCCEEDED
        repository: Any = app.repository
        models = {model.model_id: model for model in repository.list_models()}
        assert "particles-v1" in models
        registered = models["particles-v1"]
        assert registered.task is ModelTask.DETECT
        assert registered.path == run.weights_path
        assert registered.class_map == {0: "particle"}
        #: The checksum M8-T04 made this layer compute rather than ask for.
        assert registered.sha256

    def test_the_run_is_in_the_project_afterwards(self, session: SessionViewModel) -> None:
        """The provider's memory dies with the process; the project's does not
        (ADR-0084 §1). This is the window asking the right one."""
        run = train(session)

        stored = session.training_runs()
        assert [one.run_id for one in stored] == [run.run_id]
        assert stored[0].epochs_done == 3

    def test_a_second_run_is_refused_while_one_is_going(self, session: SessionViewModel) -> None:
        config = TrainingConfig(base_model="fake.pt", epochs=3, image_size_px=32)
        first = session.train(config, model_id="a", hand_drawn_only=True, val_fraction=0.25)
        settle(first)

        assert session.is_training
        assert session.train(config, model_id="b", hand_drawn_only=True, val_fraction=0.25) is None
        finish(session, first)

    def test_nothing_starts_without_a_name(self, session: SessionViewModel) -> None:
        """`start_training` refuses an empty `model_id`, because a model nothing
        can name is one no configuration can select (ADR-0050)."""
        from nanoscope.core.errors import InvalidParameterError

        config = TrainingConfig(base_model="fake.pt", epochs=2, image_size_px=32)
        job = session.train(config, model_id="", hand_drawn_only=True, val_fraction=0.0)
        assert job is not None
        assert job.wait(10.0)
        assert isinstance(job.error, InvalidParameterError)


class TestTheBuildIsAJob:
    def test_the_run_is_live_the_moment_the_build_ends(self, session: SessionViewModel) -> None:
        """The hole this task had to close, and it is not a test-timing one.

        `start_training` deliberately does not write the snapshot it returns
        (ADR-0084 §4), so nothing knew about the run until its **first epoch
        callback** — minutes, with a real trainer. In that window `is_training`
        said no, Stop was disabled, and Close Project was **enabled**: it closes
        the SQLite connection the run is writing through.
        """
        job = session.train(
            TrainingConfig(base_model="fake.pt", epochs=200, image_size_px=32),
            model_id="m",
            hand_drawn_only=True,
            val_fraction=0.0,
        )
        settle(job)

        assert session.is_training
        session.cancel_training()
        finish(session, job)

    def test_it_is_a_job_so_the_window_is_not_frozen_building_it(
        self, session: SessionViewModel
    ) -> None:
        """M8-T02's named debt, come due: preparing a scan costs 627 ms —
        measured over forty 512x512 scans, 25.1 s for the batch — so this
        cannot happen where the button is."""
        job = session.train(
            TrainingConfig(base_model="fake.pt", epochs=2, image_size_px=32),
            model_id="m",
            hand_drawn_only=True,
            val_fraction=0.0,
        )
        assert job is not None
        assert job.name == "Building the training dataset"
        finish(session, job)

    def test_a_cancelled_build_starts_no_run(self, session: SessionViewModel) -> None:
        """**Raises where an import breaks**, and this is why: a stopped build
        is a training set quietly missing the scans that came after the button,
        and nothing downstream could tell."""
        job = session.train(
            TrainingConfig(base_model="fake.pt", epochs=2, image_size_px=32),
            model_id="m",
            hand_drawn_only=True,
            val_fraction=0.0,
        )
        assert job is not None
        job.cancel()
        assert job.wait(10.0)
        QApplication.processEvents()

        assert session.training is None
        assert session.training_runs() == []


class TestWhatTheDialogShows:
    def test_a_run_with_nothing_held_out_has_no_validation_columns(
        self, session: SessionViewModel
    ) -> None:
        """ADR-0082: a `validation` block means *a held-out set existed*, not
        that a validation pass ran. With nothing held out there is no precision
        to show — and a `precision` header over empty cells is a question an
        operator spends the run wondering about."""
        dialog = TrainingDialog(session)
        train(session, val_fraction=0.0)
        dialog.show_run(session.training)

        headers = _visible_columns(dialog)
        assert "train_loss" in headers
        assert "precision" not in headers
        assert "map50" not in headers

    def test_a_run_that_held_something_out_shows_them(self, session: SessionViewModel) -> None:
        dialog = TrainingDialog(session)
        train(session, val_fraction=0.25)
        dialog.show_run(session.training)

        headers = _visible_columns(dialog)
        assert {"train_loss", "precision", "map50"} <= headers

    def test_every_epoch_is_a_row(self, session: SessionViewModel) -> None:
        dialog = TrainingDialog(session)
        run = train(session)
        dialog.show_run(run)

        assert dialog.epoch_table.rowCount() == run.epochs_done
        assert dialog.epoch_table.item(0, 0).text() == "1"

    def test_the_columns_are_the_vocabulary_and_not_a_copy(self) -> None:
        """ADR-0080 declared the metric names once, in `core`, and predicted the
        list would grow. A widget with its own copy is the one that drifts."""
        from nanoscope.core.entities.training import METRIC_NAMES
        from nanoscope.gui.dialogs.training import METRICS

        assert set(METRICS) == METRIC_NAMES

    def test_the_starting_points_come_from_the_application(self, session: SessionViewModel) -> None:
        """PROJECT_RULES §2.5: no model name is written in `gui/`, so the window
        renders what it is handed."""
        dialog = TrainingDialog(session)

        assert dialog.start_from.count() == 1
        assert dialog.start_from.currentData().base_model

    def test_a_trained_model_becomes_something_to_fine_tune(
        self, session: SessionViewModel
    ) -> None:
        train(session)
        dialog = TrainingDialog(session)

        labels = [dialog.start_from.itemText(i) for i in range(dialog.start_from.count())]
        assert any("particles-v1" in label for label in labels)


class TestAStoredRunNobodyIsRunning:
    def test_it_reads_as_interrupted_and_not_as_failed(
        self, session: SessionViewModel, app: Nanoscope
    ) -> None:
        """M8-T04 §8 decided the record — a crashed run stays `running`, because
        that is what was true when the process died. This decides the sentence,
        and the sentence is not *failed*: nobody observed a failure."""
        repository: Any = app.repository
        repository.save_training_run(
            TrainingRun(
                run_id="from-a-dead-process",
                status=TrainingStatus.RUNNING,
                dataset=DatasetSpec(root="cache/d", classes=("particle",), train_images=3),
                config=TrainingConfig(base_model="fake.pt", epochs=50, image_size_px=32),
                started_utc="2026-09-01T10:00:00+00:00",
            )
        )
        dialog = TrainingDialog(session)

        statuses = [dialog.history.item(row, 1).text() for row in range(dialog.history.rowCount())]
        assert statuses == [INTERRUPTED]
        assert "failed" not in statuses

    def test_a_live_run_is_not_called_interrupted(self, session: SessionViewModel) -> None:
        job = session.train(
            TrainingConfig(base_model="fake.pt", epochs=6, image_size_px=32),
            model_id="m",
            hand_drawn_only=True,
            val_fraction=0.0,
        )
        settle(job)
        dialog = TrainingDialog(session)

        statuses = [dialog.history.item(row, 1).text() for row in range(dialog.history.rowCount())]
        assert INTERRUPTED not in statuses
        finish(session, job)


class TestTrainingIsNotTheSameQuestionAsBusy:
    def test_the_application_stays_usable_while_a_model_trains(self, app: Nanoscope) -> None:
        """A run is hours. An application an operator cannot annotate, undo or
        export in for that long is a training appliance, not a feature."""
        window = _window(app)
        job = session_of(window).train(
            TrainingConfig(base_model="fake.pt", epochs=8, image_size_px=32),
            model_id="m",
            hand_drawn_only=True,
            val_fraction=0.0,
        )
        settle(job)
        QApplication.processEvents()

        assert session_of(window).is_training
        assert not session_of(window).is_busy
        assert window.export_all_action.isEnabled()
        assert window.import_annotations_action.isEnabled()

        #: And the three that would pull the project out from under the trainer
        #: are the ones that are not available.
        assert not window.close_action.isEnabled()
        assert not window.open_action.isEnabled()
        assert not window.new_action.isEnabled()
        finish(session_of(window), job)

    def test_the_window_can_be_opened_while_a_run_is_going(self, app: Nanoscope) -> None:
        """It is where the Stop button lives."""
        window = _window(app)
        job = session_of(window).train(
            TrainingConfig(base_model="fake.pt", epochs=8, image_size_px=32),
            model_id="m",
            hand_drawn_only=True,
            val_fraction=0.0,
        )
        settle(job)

        assert window.train_action.isEnabled()
        finish(session_of(window), job)

    def test_opening_it_twice_raises_one_window(self, app: Nanoscope) -> None:
        window = _window(app)
        window.open_training()
        first = window.training_dialog
        window.open_training()

        assert window.training_dialog is first


class TestStopping:
    def test_it_asks_and_keeps_what_was_trained(self, session: SessionViewModel) -> None:
        """ADR-0043 §3: the request is recorded and the work stops where it can
        — for a run, the next epoch boundary. Nothing is registered, because a
        cancelled run has no weights to register (ADR-0084 §5)."""
        job = session.train(
            TrainingConfig(base_model="fake.pt", epochs=200, image_size_px=32),
            model_id="stopped",
            hand_drawn_only=True,
            val_fraction=0.0,
        )
        settle(job)
        session.cancel_training()
        run = finish(session, job)

        assert run.status is TrainingStatus.CANCELLED
        assert run.epochs_done < 200
        assert not any(m.model_id == "stopped" for m in _models(session))

    def test_the_stored_record_says_it_was_cancelled(self, session: SessionViewModel) -> None:
        job = session.train(
            TrainingConfig(base_model="fake.pt", epochs=200, image_size_px=32),
            model_id="stopped",
            hand_drawn_only=True,
            val_fraction=0.0,
        )
        settle(job)
        session.cancel_training()
        finish(session, job)

        assert [one.status for one in session.training_runs()] == [TrainingStatus.CANCELLED]


class TestClosingTheWindowDuringARun:
    """The measured defect. `Nanoscope.close()` calls `jobs.shutdown(wait=True)`,
    so a six-second job made `close()` take **6.01 s** and was never asked to
    stop — with a training run that is hours of a process with no window, no
    progress and no cancel button. `wait=False` fixes nothing: it returned in
    0.00 s and the process still took the full **5.06 s** to exit, because
    `concurrent.futures` joins its threads at interpreter exit.

    So it asks, and closing cancels — which lands at an epoch boundary, which is
    all ADR-0043 ever promised.
    """

    def test_it_asks_before_walking_away(
        self, app: Nanoscope, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = _window(app)
        job = _long_run(session_of(window))
        asked: list[str] = []
        monkeypatch.setattr(
            main_window.QMessageBox,
            "question",
            lambda *args, **kwargs: asked.append(args[1]) or _CANCEL,
        )

        window.close()

        assert asked == ["A model is still training"]
        session_of(window).cancel_training()
        finish(session_of(window), job)

    def test_saying_no_keeps_the_window(
        self, app: Nanoscope, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = _window(app)
        job = _long_run(session_of(window))
        monkeypatch.setattr(main_window.QMessageBox, "question", lambda *a, **k: _CANCEL)

        assert not window.close()
        assert session_of(window).is_training

        session_of(window).cancel_training()
        finish(session_of(window), job)

    def test_closing_anyway_asks_the_run_to_stop(
        self, app: Nanoscope, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = _window(app)
        job = _long_run(session_of(window))
        monkeypatch.setattr(main_window.QMessageBox, "question", lambda *a, **k: _CLOSE)

        assert window.close()

        run = finish(session_of(window), job)
        assert run.status is TrainingStatus.CANCELLED

    def test_with_no_run_it_does_not_ask(
        self, app: Nanoscope, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A question nobody needed is one an operator learns to click through."""
        window = _window(app)
        monkeypatch.setattr(
            main_window.QMessageBox,
            "question",
            lambda *a, **k: pytest.fail("asked about a run that is not running"),
        )

        assert window.close()


def _long_run(session: SessionViewModel) -> Job:
    job = session.train(
        TrainingConfig(base_model="fake.pt", epochs=400, image_size_px=32),
        model_id="m",
        hand_drawn_only=True,
        val_fraction=0.0,
    )
    settle(job)
    assert session.is_training
    assert job is not None
    return job


def _window(app: Nanoscope) -> MainWindow:
    """A window whose session knows about the project the container opened.

    `MainWindow` constructs a session; it does not open anything, because
    opening is the composition root's (M5-T01). `refresh` is how a session
    catches up with a project that was opened around it.
    """
    window = MainWindow(app)
    window.session.refresh()
    return window


def session_of(window: MainWindow) -> SessionViewModel:
    return window.session


def _models(session: SessionViewModel) -> list[ModelDescriptor]:
    repository: Any = session._app.repository
    return list(repository.list_models())


def _visible_columns(dialog: TrainingDialog) -> set[str]:
    return {
        dialog.epoch_table.horizontalHeaderItem(column).text()
        for column in range(dialog.epoch_table.columnCount())
        if not dialog.epoch_table.isColumnHidden(column)
    }
