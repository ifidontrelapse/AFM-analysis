"""A distance somebody measured (M7-T05, ADR-0074).

The first output in this project that no algorithm produced, which is why the
roadmap's risk line for M7 says manual measurements **get their own tests**.

Two things are asserted beyond the round trip: that the length is *computed*
rather than stored, and that a scan with no scale is measured in pixels and
**says so** — the rule ADR-0025 spent a milestone on, arriving at the first
surface that produces a physical number instead of reading one.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nanoscope.app.container import Nanoscope
from nanoscope.core.entities import Ruler, RulerKind
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.science.metrology import distance_nm, distance_px
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import AnnotatePanel, ImageViewer
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")

START, END = (10.0, 10.0), (13.0, 14.0)  # 3-4-5


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name, scale in (("scaled.npy", 2.0), ("unscaled.npy", None)):
            source = tmp_path / name
            np.save(source, np.zeros((48, 48), dtype=np.float32))
            repo.import_image(source, modality=Modality.AFM, pixel_size_nm=scale)
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


def image_ids(session: SessionViewModel) -> list[int]:
    assert session.project is not None
    return [image.id for image in session.project.images]


class TestTheArithmetic:
    def test_it_is_pythagoras(self) -> None:
        assert distance_px(START, END) == pytest.approx(5.0)

    def test_two_clicks_in_one_place_measure_nothing(self) -> None:
        """Zero is a real answer, and more useful than a refusal."""
        assert distance_px(START, START) == 0.0

    def test_nanometres_need_a_scale(self) -> None:
        assert distance_nm(START, END, 2.0) == pytest.approx(10.0)
        assert distance_nm(START, END, None) is None

    def test_a_scale_that_is_not_positive_is_a_wrong_answer(self) -> None:
        """Absent is a state; zero or negative is wrong (ADR-0025's own
        distinction, made again at the first surface that *produces* a physical
        number)."""
        with pytest.raises(InvalidParameterError, match="must be positive"):
            distance_nm(START, END, 0.0)


class TestWhatIsStored:
    def test_a_line_round_trips_without_its_length(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """The length is not a column: a stored copy is a second answer waiting
        to disagree with the points it came from."""
        assert session.add_ruler(START, END, label="gap") is True

        with SqliteProjectRepository.open(project) as repository:
            stored = repository.rulers_for(image_ids(session)[0])[0]

        assert (stored.start, stored.end) == (START, END)
        assert stored.kind is RulerKind.DISTANCE
        assert not hasattr(stored, "length_px")

    def test_the_length_is_computed_in_both_units(self, session: SessionViewModel) -> None:
        session.add_ruler(START, END, label="gap")

        px, nm = session.ruler_length(session.rulers[0])

        assert (px, nm) == (pytest.approx(5.0), pytest.approx(10.0))

    def test_without_a_scale_it_is_pixels_and_says_so(self, session: SessionViewModel) -> None:
        session.select_image(image_ids(session)[1])
        panel = AnnotatePanel(session)

        session.add_ruler(START, END, label="gap")

        assert session.ruler_length(session.rulers[0])[1] is None
        assert "scale unknown" in panel.lengths.text()
        assert "5.0 px" in panel.lengths.text()

    def test_a_zero_length_line_is_discarded_quietly(self, session: SessionViewModel) -> None:
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.add_ruler(START, START, label="gap") is False
        assert session.rulers == ()
        assert said == []

    def test_the_repository_refuses_one_directly_too(self, session: SessionViewModel) -> None:
        repository = session._app.repository
        assert repository is not None

        with pytest.raises(InvalidParameterError, match="two different points"):
            repository.add_ruler(image_ids(session)[0], START, START, label="gap")

    def test_an_empty_label_is_refused(self, session: SessionViewModel) -> None:
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.add_ruler(START, END, label="  ") is False
        assert "needs a label" in said[-1]

    def test_rulers_belong_to_their_image(self, session: SessionViewModel) -> None:
        session.add_ruler(START, END, label="gap")

        session.select_image(image_ids(session)[1])

        assert session.rulers == ()


class TestUndo:
    def test_undo_removes_it_and_redo_puts_the_same_row_back(
        self, session: SessionViewModel
    ) -> None:
        session.add_ruler(START, END, label="gap")
        stored_id = session.rulers[0].id

        assert session.undo() is True
        assert session.rulers == ()

        assert session.redo() is True
        assert [one.id for one in session.rulers] == [stored_id]

    def test_the_menu_names_the_measurement(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)
        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

        window.annotate.line_drawn((START, END))

        assert "measure particle" in window.undo_action.text()


class TestTheGesture:
    def test_the_tool_suspends_panning_and_excludes_the_others(
        self, app: Nanoscope, project: Path
    ) -> None:
        window = MainWindow(app)
        window.open_project(project)
        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

        window.annotate.brush.setChecked(True)
        window.annotate.measure.setChecked(True)

        assert window.viewer.view.measuring is True
        assert window.viewer.view.painting is False
        assert not window.annotate.brush.isChecked()

    def test_a_stored_line_is_drawn(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)

        session.add_ruler(START, END, label="gap")

        assert len(viewer.view.ruler_overlay) == 1

    def test_undoing_it_takes_the_line_off_the_canvas(self, session: SessionViewModel) -> None:
        viewer = ImageViewer(session)
        session.add_ruler(START, END, label="gap")

        session.undo()

        assert viewer.view.ruler_overlay == []


class TestOneTableTwoTools:
    def test_a_profile_line_is_the_same_row_with_a_different_kind(
        self, session: SessionViewModel
    ) -> None:
        """M7-T06 reads heights along this geometry; the migration happened
        once rather than twice."""
        session.add_ruler(START, END, label="ridge", kind=RulerKind.PROFILE)

        stored = session.rulers[0]
        assert stored.kind is RulerKind.PROFILE
        assert isinstance(stored, Ruler)
