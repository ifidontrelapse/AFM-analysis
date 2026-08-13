"""An outline, stored beside its box (M7-T03, ADR-0072).

M7-T02 refused the point tool because a point has no extent and no reader.
The polygon is the other side of that argument: M7's exit criterion asks for it,
and a particle that is not a rectangle is the ordinary case in this science.

So this is the first schema change in three milestones, and what is asserted is
the shape of the compromise: **the outline is stored beside the box, not instead
of it**, so every reader that consumes boxes keeps working.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsRectItem

from nanoscope.app.container import Nanoscope
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import AnnotatePanel, ImageViewer
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")

TRIANGLE = ((10.0, 10.0), (30.0, 12.0), (20.0, 28.0))


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


class TestWhatIsStored:
    def test_the_outline_is_kept_and_the_box_is_derived(self, session: SessionViewModel) -> None:
        """A polygon and its bounding box cannot disagree, because only one of
        them is written by a caller."""
        panel = AnnotatePanel(session)

        panel.polygon_drawn(TRIANGLE)

        stored = session.annotations[0]
        assert stored.points == TRIANGLE
        assert stored.box == (10.0, 10.0, 30.0, 28.0)

    def test_a_box_annotation_still_has_no_outline(self, session: SessionViewModel) -> None:
        """`points IS NULL` means *a box, drawn as a box* — what every row
        written before this migration is."""
        panel = AnnotatePanel(session)

        panel.box_drawn((10.0, 12.0, 30.0, 34.0))

        assert session.annotations[0].points is None

    def test_it_survives_the_process(self, session: SessionViewModel, project: Path) -> None:
        panel = AnnotatePanel(session)
        panel.polygon_drawn(TRIANGLE)

        with SqliteProjectRepository.open(project) as repository:
            image_id = repository.list_images()[0].id
            reread = repository.annotations_for(image_id)[0]

        assert reread.points == TRIANGLE
        assert reread.box == (10.0, 10.0, 30.0, 28.0)

    def test_two_vertices_are_not_an_outline(self, session: SessionViewModel) -> None:
        """Two points are a line and one is the point ADR-0071 declined."""
        assert session.add_polygon(((1.0, 1.0), (5.0, 5.0)), label="line") is False
        assert session.annotations == ()

    def test_the_repository_refuses_one_directly_too(self, session: SessionViewModel) -> None:
        repository = session._app.repository
        assert repository is not None

        with pytest.raises(InvalidParameterError, match="three vertices"):
            repository.add_annotation(
                session.image_id,  # type: ignore[arg-type]
                (0.0, 0.0, 1.0, 1.0),
                label="line",
                points=[(1.0, 1.0), (5.0, 5.0)],
            )

    def test_an_empty_label_is_refused(self, session: SessionViewModel) -> None:
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.add_polygon(TRIANGLE, label="  ") is False
        assert "needs a label" in said[-1]


class TestUndoCarriesTheOutline:
    def test_redo_puts_the_polygon_back_and_not_its_box(self, session: SessionViewModel) -> None:
        """An undo that restored the box and dropped the outline would silently
        redraw the operator's work as a rectangle."""
        panel = AnnotatePanel(session)
        panel.polygon_drawn(TRIANGLE)
        stored_id = session.annotations[0].id

        assert session.undo() is True
        assert session.annotations == ()

        assert session.redo() is True
        restored = session.annotations[0]
        assert restored.id == stored_id
        assert restored.points == TRIANGLE


class TestWhatIsDrawn:
    def test_a_polygon_is_drawn_as_its_outline(self, session: SessionViewModel) -> None:
        """A polygon drawn as its bounding box is a shape nobody made."""
        viewer = ImageViewer(session)
        panel = AnnotatePanel(session)

        panel.polygon_drawn(TRIANGLE)

        item = viewer.view.annotation_overlay[0]
        assert isinstance(item, QGraphicsPolygonItem)
        assert item.polygon().count() == 3

    def test_a_box_is_still_drawn_as_a_box(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        panel = AnnotatePanel(session)

        panel.box_drawn((10.0, 12.0, 30.0, 34.0))

        assert isinstance(viewer.view.annotation_overlay[0], QGraphicsRectItem)


class TestTheGesture:
    def test_vertices_are_visible_while_the_outline_grows(self, session: SessionViewModel) -> None:
        """An outline the operator cannot see until it is finished is one they
        draw twice."""
        viewer = ImageViewer(session)
        session.select_image(session.image_id)  # type: ignore[arg-type]
        viewer.view.set_outlining(True)

        viewer.view.add_vertex(QPointF(10.0, 10.0))
        viewer.view.add_vertex(QPointF(30.0, 12.0))

        assert viewer.view._sketch is not None
        assert viewer.view._sketch.path().elementCount() == 2

    def test_closing_it_emits_the_outline_and_clears_the_sketch(
        self, session: SessionViewModel
    ) -> None:
        viewer = ImageViewer(session)
        drawn: list[tuple[tuple[float, float], ...]] = []
        viewer.view.polygon_drawn.connect(drawn.append)
        viewer.view.set_outlining(True)
        for x, y in TRIANGLE:
            viewer.view.add_vertex(QPointF(x, y))

        viewer.view.close_outline()

        assert drawn == [TRIANGLE]
        assert viewer.view._sketch is None

    def test_closing_too_early_emits_nothing(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        drawn: list[object] = []
        viewer.view.polygon_drawn.connect(drawn.append)
        viewer.view.set_outlining(True)
        viewer.view.add_vertex(QPointF(1.0, 1.0))

        viewer.view.close_outline()

        assert drawn == []

    def test_one_tool_at_a_time(self, app: Nanoscope, project: Path) -> None:
        """Two drawing modes on one canvas is a gesture that means two things."""
        window = MainWindow(app)
        window.open_project(project)
        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

        window.annotate.draw.setChecked(True)
        window.annotate.outline.setChecked(True)

        assert not window.annotate.draw.isChecked()
        assert window.viewer.view.outlining is True
        assert window.viewer.view.drawing is False
