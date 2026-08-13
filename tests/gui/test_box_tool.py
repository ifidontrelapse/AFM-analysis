"""Drawing a box, and the point that is not drawn (M7-T02, ADR-0071).

The first surface where an operator **makes** data. What is asserted:

- a drag becomes an annotation with the label the field is showing;
- **undo removes it and redo puts the same row back** — M4-T08's promise, whose
  only callers until now were its own tests;
- an empty label is refused loudly and an accidental click quietly, in that
  order;
- and there is no point tool, because the shape this project stores is a box.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QGraphicsView

from nanoscope.app.container import Nanoscope
from nanoscope.core.entities.project import AnnotationSource
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import AnnotatePanel, ImageViewer
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")

BOX = (10.0, 12.0, 30.0, 34.0)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        source = tmp_path / "one.npy"
        np.save(source, np.zeros((48, 48), dtype=np.float32))
        repo.import_image(source, modality=Modality.AFM, pixel_size_nm=2.0)
    return root


@pytest.fixture
def app(tmp_path: Path, project: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        yield container


@pytest.fixture
def session(app: Nanoscope, project: Path) -> SessionViewModel:
    model = SessionViewModel(app)
    model.open_project(project)
    assert model.project is not None
    model.select_image(model.project.images[0].id)
    return model


class TestADragBecomesAnAnnotation:
    def test_it_is_stored_with_the_operators_label(self, session: SessionViewModel) -> None:
        panel = AnnotatePanel(session)
        panel.label.setText("contamination")

        panel.box_drawn(BOX)

        assert len(session.annotations) == 1
        stored = session.annotations[0]
        assert stored.label == "contamination"
        assert stored.source is AnnotationSource.MANUAL
        assert stored.box == BOX

    def test_it_is_normalised_however_it_was_dragged(self, session: SessionViewModel) -> None:
        """Up-and-left is a drag like any other; the stored box is the one the
        repository's `CHECK` will accept (ADR-0044 §5)."""
        panel = AnnotatePanel(session)

        panel.box_drawn((30.0, 34.0, 10.0, 12.0))

        assert session.annotations[0].box == BOX

    def test_the_layer_shows_it_immediately(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        panel = AnnotatePanel(session)

        panel.box_drawn(BOX)

        assert len(viewer.view.annotation_overlay) == 1


class TestUndoAndRedo:
    def test_undo_removes_it_and_redo_puts_the_same_row_back(
        self, session: SessionViewModel
    ) -> None:
        """M4-T08's promise, with its first caller outside its own tests: a redo
        that inserted a *fresh* row would leave every command above it pointing
        at nothing."""
        panel = AnnotatePanel(session)
        panel.box_drawn(BOX)
        stored_id = session.annotations[0].id

        assert session.undo() is True
        assert session.annotations == ()

        assert session.redo() is True
        assert [one.id for one in session.annotations] == [stored_id]

    def test_the_menu_says_what_it_would_take_back(self, app: Nanoscope, project: Path) -> None:
        """ "Undo" alone makes an operator press it to find out."""
        window = MainWindow(app)
        window.open_project(project)
        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))
        assert not window.undo_action.isEnabled()

        window.annotate.box_drawn(BOX)

        assert window.undo_action.isEnabled()
        assert "particle" in window.undo_action.text()

        window._undo()

        assert not window.undo_action.isEnabled()
        assert "particle" in window.redo_action.text()

    def test_nothing_to_undo_is_not_an_error(self, session: SessionViewModel) -> None:
        assert session.undo() is False
        assert session.redo() is False


class TestWhatIsRefused:
    def test_an_empty_label_is_refused_with_a_sentence(self, session: SessionViewModel) -> None:
        """A box with no label is a rectangle (ADR-0070), and the refusal
        happens here rather than as a row saying `""`."""
        said: list[str] = []
        session.failed.connect(said.append)
        panel = AnnotatePanel(session)
        panel.label.setText("   ")

        panel.box_drawn(BOX)

        assert session.annotations == ()
        assert "needs a label" in said[-1]

    def test_an_accidental_click_is_discarded_quietly(self, session: SessionViewModel) -> None:
        """The repository refuses a zero-area box twice, but an operator who
        clicked by accident should get nothing at all, not a dialog."""
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.add_annotation((10.0, 10.0, 11.0, 11.0), label="particle") is False

        assert session.annotations == ()
        assert said == []

    def test_nothing_is_stored_without_a_selected_image(self, app: Nanoscope) -> None:
        session = SessionViewModel(app)

        assert session.add_annotation(BOX, label="particle") is False


class TestTheToolSuspendsPanning:
    def test_turning_it_on_stops_the_drag_from_panning(self, session: SessionViewModel) -> None:
        """A tool that draws *and* pans on one gesture does the wrong one half
        the time."""
        viewer = ImageViewer(session)
        assert viewer.view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag

        viewer.view.set_drawing(True)

        assert viewer.view.drawing is True
        assert viewer.view.dragMode() == QGraphicsView.DragMode.NoDrag

        viewer.view.set_drawing(False)

        assert viewer.view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag

    def test_the_window_wires_the_tool_to_the_canvas(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)
        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

        window.annotate.draw.setChecked(True)

        assert window.viewer.view.drawing is True

    def test_the_tool_is_off_without_an_image(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)

        assert not window.annotate.draw.isEnabled()


class TestThereIsNoPointTool:
    def test_every_annotation_tool_draws_a_shape_with_an_extent(
        self, session: SessionViewModel
    ) -> None:
        """ADR-0044 stores shapes with area and refuses a zero-area one twice. A
        point has no extent, so a point tool must invent one — and a "point
        size" control is that invention wearing a label. The condition for
        revisiting it is ADR-0044's own: a shape that has a reader. M7-T03 added
        outlines and M7-T04 painted masks — both shapes with area; a point is still not."""
        from PySide6.QtWidgets import QPushButton

        panel = AnnotatePanel(session)

        #: The ruler is on this list and is **not** an annotation tool: a line
        #: has no area either, which is why it got a table of its own rather
        #: than a shape (ADR-0074).
        assert [button.text() for button in panel.findChildren(QPushButton)] == [
            "Draw boxes",
            "Draw outlines",
            "Paint masks",
            "Measure distance",
        ]

    def test_a_zero_extent_annotation_cannot_be_stored_at_all(
        self, session: SessionViewModel
    ) -> None:
        """Which is what a point would be. The repository refuses it, and so
        does a `CHECK` — this asserts the refusal rather than the absence of a
        button, because the absence is the *consequence*."""
        from nanoscope.core.errors import InvalidParameterError

        repository = session._app.repository
        assert repository is not None

        with pytest.raises(InvalidParameterError):
            repository.add_annotation(
                session.image_id,  # type: ignore[arg-type]
                (10.0, 10.0, 10.0, 10.0),
                label="a point",
            )
