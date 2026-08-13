"""Moving through a project's scans (M6-T08, ADR-0068).

The workflow M6 assembles is *look at a scan, run it, read the numbers* — and the
real version is doing it to forty scans in a row. What is asserted here is the
order, the ends, and the two things that would otherwise lie: the explorer's
highlighted row, and the count in the status bar.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nanoscope.app.container import Nanoscope
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import ProjectExplorer
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")

NAMES = ("one.npy", "two.npy", "three.npy")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "P"
    with SqliteProjectRepository.create(root, "P") as repo:
        for name in NAMES:
            source = tmp_path / name
            np.save(source, np.zeros((16, 16), dtype=np.float32))
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
    return model


def image_ids(session: SessionViewModel) -> list[int]:
    assert session.project is not None
    return [image.id for image in session.project.images]


class TestTheOrderIsTheProjects:
    def test_next_and_previous_walk_it(self, session: SessionViewModel) -> None:
        ids = image_ids(session)
        session.select_image(ids[0])

        assert session.select_next() is True
        assert session.image_id == ids[1]
        assert session.select_next() is True
        assert session.image_id == ids[2]
        assert session.select_previous() is True
        assert session.image_id == ids[1]

    def test_it_does_not_wrap(self, session: SessionViewModel) -> None:
        """Wrapping takes an operator from the fortieth scan to the first
        without saying so, and *"did I look at all of them?"* is exactly the
        review that must not lie."""
        ids = image_ids(session)
        session.select_image(ids[-1])

        assert session.select_next() is False
        assert session.image_id == ids[-1]

        session.select_image(ids[0])

        assert session.select_previous() is False
        assert session.image_id == ids[0]

    def test_nothing_selected_goes_nowhere(self, session: SessionViewModel) -> None:
        assert session.select_next() is False
        assert session.image_position is None
        assert session.position_text() == ""


class TestWhereTheOperatorIs:
    def test_the_count_is_one_based_and_says_how_many(self, session: SessionViewModel) -> None:
        session.select_image(image_ids(session)[1])

        assert session.image_position == (2, 3)
        assert session.position_text() == "2 of 3"

    def test_the_window_shows_it(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)

        window.session.select_image(image_ids(window.session)[0])

        assert window.position.text() == "1 of 3"

    def test_the_actions_go_dead_at_the_ends(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)
        ids = image_ids(window.session)

        window.session.select_image(ids[0])
        assert not window.previous_action.isEnabled()
        assert window.next_action.isEnabled()

        window.session.select_image(ids[-1])
        assert window.previous_action.isEnabled()
        assert not window.next_action.isEnabled()

    def test_they_are_off_with_no_project(self, app: Nanoscope) -> None:
        window = MainWindow(app)

        assert not window.next_action.isEnabled()
        assert not window.previous_action.isEnabled()
        assert window.position.text() == ""


class TestTheExplorerFollows:
    def test_its_row_moves_with_the_selection(self, session: SessionViewModel) -> None:
        """A panel listing the images while a different one is on screen is a
        panel that lies."""
        explorer = ProjectExplorer(session)
        ids = image_ids(session)
        session.select_image(ids[0])

        session.select_next()

        assert explorer.selected_image_id == ids[1]

    def test_following_does_not_echo(self, session: SessionViewModel) -> None:
        """Setting the row must not ask the session for the selection it just
        announced — the loop M6-T05 met once on the measurements table."""
        explorer = ProjectExplorer(session)
        session.select_image(image_ids(session)[0])
        announced: list[object] = []
        session.image_changed.connect(announced.append)

        session.select_next()

        assert len(announced) == 1
        assert explorer.selected_image_id == session.image_id

    def test_a_click_in_the_explorer_still_drives(self, session: SessionViewModel) -> None:
        explorer = ProjectExplorer(session)

        explorer.tree.setCurrentItem(explorer.tree.topLevelItem(2))

        assert session.image_id == image_ids(session)[2]
        assert session.position_text() == "3 of 3"
