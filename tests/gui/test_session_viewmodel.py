"""The session, driven without a single widget (M5-T06, ADR-0057).

That is the point of the file: every test here builds a `SessionViewModel` and
**no `QWidget` at all**. A viewmodel that needs a window to be tested is one that
has a window inside it, and the confirmation dialogs, the status bar and the
docks are exactly what this layer exists to be separable from.

What is asserted is the contract the panels rely on: intent goes in as a method
call, state comes out as a signal, and a refusal is a sentence rather than an
exception crossing into a widget.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nanoscope.app.container import Nanoscope
from nanoscope.application.use_cases.display import DisplayImage
from nanoscope.core.values import Modality
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")

BOX = (10.0, 10.0, 30.0, 30.0)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Two scans: one with a known scale, one without."""
    root = tmp_path / "Gold on mica"
    with SqliteProjectRepository.create(root, "Gold on mica") as repo:
        for name, scale in (("monday.npy", 2.5), ("tuesday.npy", None)):
            source = tmp_path / name
            np.save(source, np.zeros((16, 16), dtype=np.float32))
            repo.import_image(source, modality=Modality.AFM, pixel_size_nm=scale)
    return root


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        yield container


@pytest.fixture
def session(app: Nanoscope) -> SessionViewModel:
    return SessionViewModel(app)


def image_ids(session: SessionViewModel) -> list[int]:
    assert session.project is not None
    return [image.id for image in session.project.images]


class TestTheProject:
    def test_opening_one_announces_it(self, session: SessionViewModel, project: Path) -> None:
        announced: list[object] = []
        session.project_changed.connect(announced.append)

        opened = session.open_project(project)

        assert opened is not None
        assert session.project is opened
        assert announced == [opened]

    def test_a_refusal_is_a_sentence_and_not_an_exception(
        self, session: SessionViewModel, tmp_path: Path
    ) -> None:
        """Nothing above this layer catches `NanoscopeError`; the window shows
        what it is told (ADR-0030)."""
        said: list[str] = []
        session.failed.connect(said.append)

        assert session.open_project(tmp_path / "not-a-project") is None

        assert session.project is None
        assert "not a project directory" in said[0]

    def test_closing_empties_the_session(self, session: SessionViewModel, project: Path) -> None:
        session.open_project(project)
        session.select_image(image_ids(session)[0])
        announced: list[object] = []
        session.project_changed.connect(announced.append)

        session.close_project()

        assert session.project is None
        assert session.image is None and session.image_id is None
        assert announced == [None]

    def test_opening_a_second_project_drops_the_first_selection(
        self, session: SessionViewModel, project: Path, tmp_path: Path
    ) -> None:
        """Ids are per-project: image 3 of the old one is not image 3 of the
        new one, and a panel still showing it would be showing a stranger."""
        session.open_project(project)
        session.select_image(image_ids(session)[0])
        other = tmp_path / "Empty"
        SqliteProjectRepository.create(other, "Empty").close()

        session.open_project(other)

        assert session.image is None and session.image_id is None


class TestSelectingAnImage:
    def test_it_loads_once_and_announces_the_array(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """One load per selection, for every panel that shows one — the reason
        this layer exists rather than each panel reading the file."""
        session.open_project(project)
        announced: list[object] = []
        session.image_changed.connect(announced.append)

        assert session.select_image(image_ids(session)[0]) is True

        assert isinstance(session.image, DisplayImage)
        assert announced == [session.image]
        assert session.image.pixel_size_nm == 2.5

    def test_an_unreadable_file_clears_the_view_and_says_why(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """A panel left showing the previous scan under a new selection is a
        panel that lies quietly."""
        session.open_project(project)
        session.select_image(image_ids(session)[0])
        (project / "images" / "tuesday.npy").unlink()
        announced: list[object] = []
        said: list[str] = []
        session.image_changed.connect(announced.append)
        session.failed.connect(said.append)

        assert session.select_image(image_ids(session)[1]) is False

        assert session.image is None
        assert announced == [None]
        assert said

    def test_a_failed_load_is_still_a_selection(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """A scan whose file is missing is still the row the operator clicked,
        and removing it is the likeliest thing they want to do next."""
        session.open_project(project)
        (project / "images" / "monday.npy").unlink()

        session.select_image(image_ids(session)[0])

        assert session.image is None
        assert session.image_id == image_ids(session)[0]

    def test_selecting_without_a_project_does_nothing(self, session: SessionViewModel) -> None:
        assert session.select_image(1) is False


class TestRemovingAnImage:
    def test_it_counts_what_would_be_lost_without_deciding(
        self, app: Nanoscope, session: SessionViewModel, project: Path
    ) -> None:
        """The count is the viewmodel's; whether to ask, and in what words, is
        the panel's (ADR-0055 stays where it was written)."""
        session.open_project(project)
        assert app.repository is not None
        image_id = image_ids(session)[0]
        for _ in range(3):
            app.repository.add_annotation(image_id, BOX, label="particle")

        assert session.annotation_count(image_id) == 3
        assert session.annotation_count(image_ids(session)[1]) == 0

    def test_removing_the_selected_image_clears_it(
        self, session: SessionViewModel, project: Path
    ) -> None:
        session.open_project(project)
        image_id = image_ids(session)[0]
        session.select_image(image_id)

        assert session.remove_image(image_id) is True

        assert session.image is None and session.image_id is None
        assert [image.display_name for image in session.project.images] == ["tuesday.npy"]  # type: ignore[union-attr]

    def test_removing_another_image_leaves_the_view_alone(
        self, session: SessionViewModel, project: Path
    ) -> None:
        session.open_project(project)
        looking_at = image_ids(session)[0]
        session.select_image(looking_at)

        session.remove_image(image_ids(session)[1])

        assert session.image_id == looking_at
        assert session.image is not None

    def test_the_project_is_re_read_rather_than_edited(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """The integrity report is part of what the panels show, and the file
        left behind has just become untracked (ADR-0040)."""
        session.open_project(project)

        session.remove_image(image_ids(session)[0])

        assert session.project is not None
        assert session.project.integrity.untracked_files == ("images/monday.npy",)

    def test_removing_without_a_project_does_nothing(self, session: SessionViewModel) -> None:
        assert session.remove_image(1) is False
        assert session.annotation_count(1) == 0


class TestTheRecordLookup:
    def test_it_answers_from_the_open_project(
        self, session: SessionViewModel, project: Path
    ) -> None:
        """The panels need a name and a path for a sentence, and the project
        they were built from already holds both — no second query."""
        session.open_project(project)

        record = session.image_record(image_ids(session)[0])

        assert record is not None
        assert record.display_name == "monday.npy"

    def test_an_unknown_id_is_none_rather_than_an_error(
        self, session: SessionViewModel, project: Path
    ) -> None:
        session.open_project(project)

        assert session.image_record(9_999) is None
        assert SessionViewModel(session._app).image_record(1) is None
