"""Undo/redo through every tool (M7-T08, ADR-0077).

Every tool since M7-T02 already goes through the command stack, so this task is
an **audit** — and these are the three gaps it found, each with the task that
wrote it down:

1. Nothing told the window that the **history** moved. M7-T02 read that off
   `annotations_changed` and said it would hold only while every command mutated
   annotations; M7-T05's ruler ended that, and a third signal would have been the
   same mistake again.
2. Adopting forty detections cost forty undos.
3. Undo could edit a scan nobody was looking at, and the screen would not change.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.application.jobs import Job
from nanoscope.core.entities import PipelineConfig, RulerKind
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def phantom(size: int = 48) -> np.ndarray:
    """Two Gaussian particles, so a detect run has something to find."""
    rng = np.random.default_rng(11)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    height = rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    for cy, cx in ((14, 14), (32, 30)):
        height += 5.0 * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 10.0))
    return height.astype(np.float32)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Two scans, because the history is per project and the layers are per scan."""
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name in ("first.npy", "second.npy"):
            source = tmp_path / name
            np.save(source, phantom())
            repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
    return root


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        yield container


@pytest.fixture
def session(app: Nanoscope, project: Path) -> SessionViewModel:
    model = SessionViewModel(app)
    model.open_project(project)
    assert model.project is not None
    model.select_image(model.project.images[0].id)
    return model


def image_ids(session: SessionViewModel) -> list[int]:
    assert session.project is not None
    return [image.id for image in session.project.images]


def analyse(session: SessionViewModel) -> None:
    job = session.detect(PipelineConfig(detector="log", mode="detect"))
    assert isinstance(job, Job)
    assert job.wait(60.0)
    QApplication.processEvents()


#: Every tool in M7, as the one call the panel makes. The point of the list is
#: that it is a list: a tool added without an entry here is a tool nobody checked
#: goes through the stack.
TOOLS: dict[str, Callable[[SessionViewModel], object]] = {
    "box": lambda s: s.add_annotation((5.0, 5.0, 20.0, 20.0), label="blob"),
    "outline": lambda s: s.add_polygon(((5.0, 5.0), (20.0, 6.0), (12.0, 20.0)), label="blob"),
    "mask": lambda s: s.add_mask(_painted(), label="blob"),
    "ruler": lambda s: s.add_ruler((5.0, 5.0), (8.0, 9.0), label="across"),
    "profile": lambda s: s.add_ruler((5.0, 5.0), (8.0, 9.0), label="along", kind=RulerKind.PROFILE),
}


def _painted() -> np.ndarray:
    mask = np.zeros((48, 48), dtype=bool)
    mask[10:20, 10:20] = True
    return mask


class TestTheHistorySaysSo:
    @pytest.mark.parametrize("tool", list(TOOLS), ids=list(TOOLS))
    def test_every_tool_is_one_step_that_announces_itself(
        self, session: SessionViewModel, tool: str
    ) -> None:
        """One gesture, one entry on the history, one `history_changed`."""
        moved: list[int] = []
        session.history_changed.connect(lambda: moved.append(1))

        assert TOOLS[tool](session) is True

        assert len(moved) == 1
        assert session.undo_label is not None
        assert session.undo() is True
        assert session.undo_label is None

    def test_the_window_names_what_undo_would_take_back(
        self, app: Nanoscope, project: Path
    ) -> None:
        """M7-T02 wrote the labels; what this asserts is that the *menu* hears
        about the history from the history."""
        window = MainWindow(app)
        window.session.open_project(project)
        window.session.select_image(image_ids(window.session)[0])
        assert not window.undo_action.isEnabled()

        window.session.add_annotation((5.0, 5.0, 20.0, 20.0), label="blob")

        assert window.undo_action.isEnabled()
        assert window.undo_action.text() == "&Undo add blob"
        assert not window.redo_action.isEnabled()

        window.undo_action.trigger()

        assert not window.undo_action.isEnabled()
        assert window.undo_action.text() == "&Undo"
        assert window.redo_action.text() == "&Redo add blob"

        window.redo_action.trigger()

        assert window.undo_action.text() == "&Undo add blob"

    def test_a_command_that_touches_no_annotation_still_moves_the_menu(
        self, app: Nanoscope, project: Path
    ) -> None:
        """The gap this task closed: a ruler is not an annotation, and until
        M7-T05 the Undo item was labelled by a signal about annotations."""
        window = MainWindow(app)
        window.session.open_project(project)
        window.session.select_image(image_ids(window.session)[0])

        window.session.add_ruler((5.0, 5.0), (8.0, 9.0), label="across")

        assert window.undo_action.text() == "&Undo measure across"

    def test_closing_the_project_empties_the_history_and_the_menu(
        self, app: Nanoscope, project: Path
    ) -> None:
        """Undo is a session (ADR-0045), and a menu offering to undo an edit in a
        closed project is offering something that cannot happen."""
        window = MainWindow(app)
        window.session.open_project(project)
        window.session.select_image(image_ids(window.session)[0])
        window.session.add_annotation((5.0, 5.0, 20.0, 20.0), label="blob")

        window.session.close_project()

        assert window.session.undo_label is None
        assert not window.undo_action.isEnabled()


class TestOneGestureIsOneUndo:
    def test_adopting_every_detection_undoes_in_one_step(self, session: SessionViewModel) -> None:
        """Adoption is one click (ADR-0076 §3), so taking it back is one
        `Ctrl+Z` — not one per detection."""
        analyse(session)
        run = session.run
        assert run is not None and len(run.detections) > 1

        adopted = session.adopt_all_detections(label="particle")

        assert adopted == len(run.detections)
        assert session.undo_label == f"adopt {len(run.detections)} detection(s)"

        assert session.undo() is True

        assert session.annotations == ()
        assert session.undo_label is None

    def test_redo_puts_the_same_rows_back(self, session: SessionViewModel) -> None:
        """ADR-0045's promise, for a batch: the ids survive, or every command
        above this one points at nothing."""
        analyse(session)
        session.adopt_all_detections(label="particle")
        before = [one.id for one in session.annotations]

        session.undo()
        assert session.redo() is True

        assert [one.id for one in session.annotations] == before

    def test_an_empty_label_refuses_the_whole_batch(self, session: SessionViewModel) -> None:
        analyse(session)
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.adopt_all_detections(label="  ") == 0

        assert "needs a label" in said[-1]
        assert session.annotations == ()
        assert session.undo_label is None


class TestUndoFollowsTheWork:
    def test_it_selects_the_scan_whose_work_it_took_back(self, session: SessionViewModel) -> None:
        """Otherwise undoing an edit made on another scan removes a row nobody
        can see, and the window does not change (M7-T08)."""
        first, second = image_ids(session)
        session.add_annotation((5.0, 5.0, 20.0, 20.0), label="blob")
        annotation_id = session.annotations[0].id
        session.select_image(second)
        assert session.annotations == ()

        assert session.undo() is True

        assert session.image_id == first
        assert all(one.id != annotation_id for one in session.annotations)

    def test_a_ruler_undone_from_another_scan_comes_off_that_scans_canvas(
        self, session: SessionViewModel
    ) -> None:
        first, second = image_ids(session)
        session.add_ruler((5.0, 5.0), (8.0, 9.0), label="across")
        session.select_image(second)

        assert session.undo() is True

        assert session.image_id == first
        assert session.rulers == ()

    def test_an_edit_on_the_selected_scan_does_not_reload_it(
        self, session: SessionViewModel
    ) -> None:
        """The scan is only re-selected when the history moved somewhere else:
        reloading the array to undo a box is a disk read for nothing."""
        session.add_annotation((5.0, 5.0, 20.0, 20.0), label="blob")
        loaded: list[object] = []
        session.image_changed.connect(loaded.append)

        assert session.undo() is True

        assert loaded == []
        assert session.annotations == ()
