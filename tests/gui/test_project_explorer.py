"""The panel, and the confirmation ADR-0044 asked for by name (M5-T04, ADR-0055).

The centre of this file is `TestRemovingAnImage`. ADR-0044 decided that
annotations cascade when an image is removed, and ended on an obligation
addressed to this task:

> *`annotations_for` exists to be counted **before** the deletion, by a
> confirmation dialog that can say "this image has 12 annotations".*

So the tests here assert the **count**, that cancelling changes nothing, and
that an image with nothing to lose is removed without a dialog at all — because
a confirmation that always appears is one nobody reads by the third time.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from nanoscope.app.container import Nanoscope
from nanoscope.application.settings import COLORMAP_SETTING
from nanoscope.core.values import Modality
from nanoscope.gui.panels import ProjectExplorer
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic_spm import write_spm, z_field

pytestmark = pytest.mark.usefixtures("qt_app")

BOX = (10.0, 10.0, 30.0, 30.0)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Two images: one with a known scale, one without."""
    root = tmp_path / "Gold on mica"
    with SqliteProjectRepository.create(root, "Gold on mica") as repo:
        for name, scale in (("monday.spm", 1.95), ("tuesday.npy", None)):
            (repo.root / "images" / name).write_bytes(b"AFM")
            repo.add_image(f"images/{name}", modality=Modality.AFM, pixel_size_nm=scale)
    return root


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        yield container


@pytest.fixture
def session(app: Nanoscope, project: Path) -> SessionViewModel:
    """The panel's only collaborator (M5-T06): it holds no container."""
    model = SessionViewModel(app)
    model.open_project(project)
    return model


@pytest.fixture
def explorer(session: SessionViewModel) -> ProjectExplorer:
    return ProjectExplorer(session)


def rows(panel: ProjectExplorer) -> list[str]:
    tree = panel.tree
    return [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]


class TestWhatItShows:
    def test_it_lists_the_projects_images(self, explorer: ProjectExplorer) -> None:
        assert rows(explorer) == ["monday.spm", "tuesday.npy"]

    def test_it_shows_the_scale_and_says_when_there_is_none(
        self, explorer: ProjectExplorer
    ) -> None:
        """An unknown scale is a state, not a blank cell (ADR-0025, one layer
        out)."""
        tree = explorer.tree

        assert tree.topLevelItem(0).text(1) == "1.95 nm/px"
        assert tree.topLevelItem(1).text(1) == "scale unknown"

    def test_a_missing_file_is_marked(self, session: SessionViewModel, project: Path) -> None:
        """A panel that lists an image whose file is gone without saying so is a
        panel that lies quietly. The report is already in hand (ADR-0040)."""
        (project / "images" / "monday.spm").unlink()
        session.refresh()

        panel = ProjectExplorer(session)

        assert "file missing" in rows(panel)[0]
        assert "file missing" not in rows(panel)[1]

    def test_closing_the_project_empties_it(
        self, explorer: ProjectExplorer, session: SessionViewModel
    ) -> None:
        """The panel is not told to empty itself; it hears that the project
        closed and rebuilds from that."""
        session.close_project()

        assert rows(explorer) == []

    def test_selecting_a_row_tells_the_session(
        self, explorer: ProjectExplorer, session: SessionViewModel
    ) -> None:
        """A selection is an *intent*: the panel hands it to the viewmodel, and
        the viewer hears about it from there rather than from this widget
        (ADR-0057). The file is a stub, so it does not load — and the selection
        stands anyway, which is the distinction `image_id` exists for."""
        explorer.tree.setCurrentItem(explorer.tree.topLevelItem(1))

        assert session.image_id == explorer.selected_image_id


class TestRemovingAnImage:
    def test_it_asks_first_and_says_how_many_annotations_would_go(
        self,
        app: Nanoscope,
        session: SessionViewModel,
        explorer: ProjectExplorer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ADR-0044's obligation, discharged: the dialog carries the **count**,
        not "are you sure?"."""
        assert app.repository is not None
        image = app.repository.list_images()[0]
        for _ in range(3):
            app.repository.add_annotation(image.id, BOX, label="particle")
        session.refresh()
        explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))

        asked: list[str] = []
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: (
                asked.append(args[2]),
                QMessageBox.StandardButton.Yes,
            )[1],
        )

        assert explorer.remove_selected() is True

        assert "3 annotation(s)" in asked[0]
        assert "cannot be recomputed" in asked[0]
        assert "images/monday.spm" in asked[0]

    def test_cancelling_changes_nothing(
        self,
        app: Nanoscope,
        session: SessionViewModel,
        explorer: ProjectExplorer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert app.repository is not None
        image = app.repository.list_images()[0]
        app.repository.add_annotation(image.id, BOX, label="particle")
        session.refresh()
        explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
        )

        assert explorer.remove_selected() is False

        assert len(app.repository.list_images()) == 2
        assert len(app.repository.annotations_for(image.id)) == 1

    def test_an_image_with_nothing_to_lose_is_removed_without_a_dialog(
        self, app: Nanoscope, explorer: ProjectExplorer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A confirmation that always appears is one nobody reads by the third
        time — and then the one that mattered is clicked through as well."""
        asked: list[object] = []
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a), None)[1])
        explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))

        assert explorer.remove_selected() is True

        assert asked == []
        assert rows(explorer) == ["tuesday.npy"]

    def test_the_file_stays_where_it_was(
        self, app: Nanoscope, explorer: ProjectExplorer, project: Path
    ) -> None:
        """Forgetting a scan and deleting it are different decisions, and this
        layer does not get to make the second one (ADR-0040)."""
        explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))

        explorer.remove_selected()

        assert (project / "images" / "monday.spm").is_file()

    def test_removing_nothing_selected_does_nothing(self, explorer: ProjectExplorer) -> None:
        explorer.tree.clearSelection()

        assert explorer.remove_selected() is False

    def test_the_panel_refreshes_itself_afterwards(
        self, app: Nanoscope, explorer: ProjectExplorer
    ) -> None:
        """Rebuilt from the repository rather than edited in place, because the
        integrity report is part of what is shown — and the removed image's file
        is now untracked."""
        explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0))

        explorer.remove_selected()

        assert rows(explorer) == ["tuesday.npy"]
        assert app.repository is not None
        assert app.repository.check_integrity().untracked_files == ("images/monday.spm",)


# ── the pictures (2026-09-02) ─────────────────────────────────────────────────


@pytest.fixture
def readable(tmp_path: Path) -> Path:
    """A project whose files are real: a Nanoscope scan, an array, and a stub.

    The fixture above writes `b"AFM"` into `images/`, which is enough for every
    test about *rows* and nothing at all for a test about *pictures*.
    """
    root = tmp_path / "Readable"
    with SqliteProjectRepository.create(root, "Readable") as repo:
        repo.import_image(write_spm(tmp_path, z_field(), name="scan.000"), modality=Modality.AFM)
        flat = tmp_path / "flat.npy"
        np.save(flat, np.linspace(0, 1, 256).reshape(16, 16).astype(np.float32))
        repo.import_image(flat, modality=Modality.AFM, pixel_size_nm=2.0)
        broken = tmp_path / "half-a-scan.001"
        broken.write_bytes(b"\x1a" * 32)
        repo.import_image(broken, modality=Modality.AFM)
    return root


@pytest.fixture
def pictures(app: Nanoscope, readable: Path) -> ProjectExplorer:
    model = SessionViewModel(app)
    model.open_project(readable)
    return ProjectExplorer(model)


def drain(panel: ProjectExplorer) -> int:
    """Draw every queued thumbnail, and say how many turns it took.

    The panel schedules itself through `QTimer.singleShot(0, …)`, which needs an
    event loop; a test that wants the end state steps the pump instead of
    running one.
    """
    turns = 1
    while panel.draw_next_thumbnail():
        turns += 1
    return turns


class TestTheThumbnails:
    def test_every_readable_row_gets_a_picture(self, pictures: ProjectExplorer) -> None:
        drain(pictures)

        icons = [pictures.tree.topLevelItem(i).icon(0) for i in range(3)]
        assert [icon.isNull() for icon in icons] == [False, False, True]

    def test_a_file_that_cannot_be_read_keeps_its_row(self, pictures: ProjectExplorer) -> None:
        """Unreadable is not empty: the name, the scale and the tooltip stay,
        and the refusal is the viewer's sentence to say (ADR-0030)."""
        drain(pictures)

        row = pictures.tree.topLevelItem(2)
        assert row.text(0) == "half-a-scan.001"
        assert row.icon(0).isNull()

    def test_they_are_drawn_one_per_turn_and_not_all_at_once(
        self, pictures: ProjectExplorer
    ) -> None:
        """The whole reason there is a queue: a list that reads forty files
        before it appears is a window that hangs on open."""
        assert pictures.tree.topLevelItem(0).icon(0).isNull()

        pictures.draw_next_thumbnail()

        assert not pictures.tree.topLevelItem(0).icon(0).isNull()
        assert pictures.tree.topLevelItem(1).icon(0).isNull()

    def test_the_queue_ends(self, pictures: ProjectExplorer) -> None:
        assert drain(pictures) == 3
        assert pictures.draw_next_thumbnail() is False

    def test_a_new_project_abandons_the_old_queue(
        self, pictures: ProjectExplorer, project: Path
    ) -> None:
        """Opening a second project must not finish drawing the first — the
        rows those ids belong to are gone."""
        pictures._session.open_project(project)

        drain(pictures)

        assert rows(pictures) == ["monday.spm", "tuesday.npy"]
        assert all(pictures.tree.topLevelItem(i).icon(0).isNull() for i in range(2))

    def test_a_missing_file_is_never_asked_for(
        self, pictures: ProjectExplorer, readable: Path
    ) -> None:
        """ADR-0040 already reported it; a read per missing row would only
        rediscover the report one file at a time."""
        (readable / "images" / "scan.000").unlink()
        pictures._session.refresh()

        drain(pictures)

        assert "file missing" in pictures.tree.topLevelItem(0).text(0)
        assert pictures.tree.topLevelItem(0).icon(0).isNull()

    def test_the_picture_uses_the_operators_default_colormap(
        self, pictures: ProjectExplorer
    ) -> None:
        """The same preference the viewer opens a scan with (M5-T09), so a row
        and the canvas above it are the same picture."""
        pictures._session.remember(COLORMAP_SETTING, "gray")

        assert pictures._colormap() == "gray"

    def test_a_stored_colormap_this_version_does_not_offer_is_ignored(
        self, pictures: ProjectExplorer
    ) -> None:
        """`render` would refuse it, and a hand-edited settings file is not
        worth an empty list."""
        pictures._session.remember(COLORMAP_SETTING, "chartreuse")

        assert pictures._colormap() == "afmhot"
