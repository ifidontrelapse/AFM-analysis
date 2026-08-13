"""A painted mask, and where it lives (M7-T04, ADR-0073).

The third shape, and the first that is not a handful of numbers. PROJECT_RULES §5
decided where it goes before this task existed — *"no mask bitmaps in the
database; masks are files, the database stores paths"* — so what is asserted here
is the storage as much as the brush.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from nanoscope.app.container import Nanoscope
from nanoscope.core.errors import InvalidParameterError, MissingFileError
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import AnnotatePanel, ImageViewer
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


def blob(size: int = 48) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[10:20, 12:24] = True
    return mask


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


class TestWhereTheMaskGoes:
    def test_it_is_a_file_and_the_row_points_at_it(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """PROJECT_RULES §5, and ADR-0042 made the same call for measurement
        tables: an array in a database column is a blob nothing can read without
        this application."""
        panel = AnnotatePanel(session)

        panel.mask_painted(blob())

        stored = session.annotations[0]
        assert stored.mask_path is not None
        assert stored.mask_path.startswith("annotations/")
        assert (project / stored.mask_path).is_file()

    def test_the_box_is_derived_from_the_painted_pixels(self, session: SessionViewModel) -> None:
        """The same rule as an outline's: whatever a reader wants as a box, the
        repository computes from what was actually drawn."""
        panel = AnnotatePanel(session)

        panel.mask_painted(blob())

        assert session.annotations[0].box == (12.0, 10.0, 24.0, 20.0)

    def test_it_reads_back_as_what_was_painted(self, session: SessionViewModel) -> None:
        panel = AnnotatePanel(session)
        painted = blob()

        panel.mask_painted(painted)

        assert np.array_equal(session.mask_of(session.annotations[0]), painted)

    def test_a_stroke_that_painted_nothing_stores_nothing(self, session: SessionViewModel) -> None:
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.add_mask(np.zeros((48, 48), dtype=bool), label="particle") is False

        assert session.annotations == ()
        assert said == []

    def test_the_repository_refuses_an_empty_one_directly(self, session: SessionViewModel) -> None:
        repository = session._app.repository
        assert repository is not None

        with pytest.raises(InvalidParameterError, match="nothing was painted"):
            repository.add_annotation(
                session.image_id,  # type: ignore[arg-type]
                (0.0, 0.0, 1.0, 1.0),
                label="empty",
                mask=np.zeros((8, 8), dtype=bool),
            )

    def test_an_empty_label_is_refused(self, session: SessionViewModel) -> None:
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.add_mask(blob(), label=" ") is False
        assert "needs a label" in said[-1]

    def test_a_missing_file_is_a_refusal_not_an_empty_mask(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """An empty mask would read as "the operator painted nothing", which is
        a different statement (ADR-0040)."""
        panel = AnnotatePanel(session)
        panel.mask_painted(blob())
        stored = session.annotations[0]
        assert stored.mask_path is not None
        (project / stored.mask_path).unlink()
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.mask_of(stored) is None
        assert said

        repository = session._app.repository
        assert repository is not None
        with pytest.raises(MissingFileError):
            repository.mask_of(stored)


class TestUndoLeavesTheFile:
    def test_undo_removes_the_row_and_keeps_the_file(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """Forgetting a thing and deleting a file are different decisions
        (ADR-0040, third application)."""
        panel = AnnotatePanel(session)
        panel.mask_painted(blob())
        path = project / session.annotations[0].mask_path  # type: ignore[operator]

        assert session.undo() is True

        assert session.annotations == ()
        assert path.is_file()

    def test_redo_points_the_restored_row_at_the_same_file(self, session: SessionViewModel) -> None:
        panel = AnnotatePanel(session)
        panel.mask_painted(blob())
        before = session.annotations[0]
        session.undo()

        assert session.redo() is True

        restored = session.annotations[0]
        assert (restored.id, restored.mask_path) == (before.id, before.mask_path)
        assert np.array_equal(session.mask_of(restored), blob())


class TestThePainting:
    def test_a_dab_paints_a_disc_and_never_the_scan(self, session: SessionViewModel) -> None:
        """The brush paints into a mask of its own: a tool that edited the data
        an operator is measuring would be the worst version of this feature."""
        viewer = ImageViewer(session)
        before = viewer.view._item.pixmap().toImage()
        viewer.view.set_painting(True, brush_px=3)

        viewer.view.paint_at(QPointF(20.0, 20.0))

        assert viewer.view._stroke is not None
        assert viewer.view._stroke[20, 20]
        assert not viewer.view._stroke[0, 0]
        assert viewer.view._item.pixmap().toImage() == before

    def test_lifting_the_brush_hands_the_stroke_over(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        painted: list[np.ndarray] = []
        viewer.view.mask_painted.connect(painted.append)
        viewer.view.set_painting(True, brush_px=2)
        viewer.view.paint_at(QPointF(10.0, 10.0))

        viewer.view.finish_stroke()

        assert len(painted) == 1
        assert painted[0][10, 10]
        assert viewer.view._stroke is None

    def test_an_empty_stroke_hands_over_nothing(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        painted: list[object] = []
        viewer.view.mask_painted.connect(painted.append)
        viewer.view.set_painting(True)

        viewer.view.finish_stroke()

        assert painted == []

    def test_the_layer_draws_it_as_an_outline(self, session: SessionViewModel) -> None:
        """Outlined like every other mask: a fill hides the pixels it describes
        (ADR-0064 §6)."""
        viewer = ImageViewer(session)
        panel = AnnotatePanel(session)

        panel.mask_painted(blob())

        assert len(viewer.view.painted_overlay) == 1

    def test_one_tool_at_a_time(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)
        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

        window.annotate.outline.setChecked(True)
        window.annotate.brush.setChecked(True)

        assert not window.annotate.outline.isChecked()
        assert window.viewer.view.painting is True
        assert window.viewer.view.outlining is False

    def test_the_brush_size_reaches_the_canvas(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)
        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

        window.annotate.brush_size.setValue(17)

        assert window.viewer.view._brush_px == 17
