"""The preferences an operator can state (M5-T09, ADR-0060).

Two things are asserted that are not "the combo has the right items":

- **every row round-trips into the store an existing reader already uses** —
  `select_device` for the device, the viewer for the colormap, `app/main.py` for
  the level. A settings dialog whose rows nothing reads is a preferences file
  with a nicer front end.
- **an open project that overrides a key is said out loud**, which is the
  sentence `Settings.scope_of` was written for in M4-T10 and has had no caller
  since.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from nanoscope.app.container import Nanoscope
from nanoscope.application.settings import (
    COLORMAP_SETTING,
    DEVICE_SETTING,
    LOG_LEVEL_SETTING,
    Scope,
)
from nanoscope.application.use_cases.display import COLORMAPS
from nanoscope.gui.dialogs.settings import AUTOMATIC, SettingsDialog
from nanoscope.gui.panels import ImageViewer
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        yield container


@pytest.fixture
def session(app: Nanoscope) -> SessionViewModel:
    return SessionViewModel(app)


@pytest.fixture(autouse=True)
def restore_level() -> Iterator[None]:
    """The dialog applies the level to the running process, which is the point —
    and would otherwise leak into every test that runs after it."""
    before = logging.getLogger().level
    yield
    logging.getLogger().setLevel(before)


class TestWhatItOffers:
    def test_the_device_list_is_this_machine(self, session: SessionViewModel) -> None:
        """Not the four the enum names, three of which would fail on any given
        computer (ADR-0049)."""
        dialog = SettingsDialog(session)

        offered = [dialog.device.itemText(i) for i in range(dialog.device.count())]

        assert offered[0] == AUTOMATIC
        assert len(offered) == 1 + len(session.devices())
        assert any(device.name in text for device in session.devices() for text in offered)

    def test_automatic_is_the_default_and_stores_nothing_of_its_own(
        self, session: SessionViewModel, app: Nanoscope
    ) -> None:
        """`None` is what `select_device` reads as *let the policy decide* — the
        answer that keeps working when the operator changes machines."""
        dialog = SettingsDialog(session)

        dialog.apply()

        assert app.settings.get(DEVICE_SETTING) is None
        assert app.select_device()  # the policy still resolves something

    def test_it_opens_on_what_is_stored(self, session: SessionViewModel) -> None:
        session.remember(COLORMAP_SETTING, "viridis")
        session.remember(LOG_LEVEL_SETTING, logging.WARNING)

        dialog = SettingsDialog(session)

        assert dialog.colormap.currentText() == "viridis"
        assert dialog.level.currentData() == logging.WARNING

    def test_a_stored_value_this_build_does_not_offer_is_ignored(
        self, session: SessionViewModel
    ) -> None:
        """A settings file describing another machine — an unplugged GPU, a
        level from a newer build — must not stop the dialog opening."""
        session.remember(DEVICE_SETTING, "quantum")

        dialog = SettingsDialog(session)

        assert dialog.device.currentText() == AUTOMATIC


class TestWhatItWrites:
    def test_every_row_reaches_the_store_a_reader_uses(
        self, session: SessionViewModel, app: Nanoscope
    ) -> None:
        dialog = SettingsDialog(session)
        dialog.device.setCurrentIndex(1)  # the first real device
        dialog.colormap.setCurrentText("magma")
        dialog.level.setCurrentIndex(0)  # DEBUG

        dialog.apply()

        assert app.settings.get(DEVICE_SETTING) == str(session.devices()[0].kind)
        assert app.settings.get(COLORMAP_SETTING) == "magma"
        assert app.settings.get(LOG_LEVEL_SETTING) == logging.DEBUG

    def test_the_log_level_applies_to_the_running_process(self, session: SessionViewModel) -> None:
        """Now, not at the next start: an operator setting DEBUG is about to
        reproduce something."""
        dialog = SettingsDialog(session)
        dialog.level.setCurrentIndex(0)

        dialog.apply()

        assert logging.getLogger().level == logging.DEBUG

    def test_cancelling_writes_nothing(self, session: SessionViewModel, app: Nanoscope) -> None:
        dialog = SettingsDialog(session)
        dialog.colormap.setCurrentText("bone")

        dialog.reject()

        assert app.settings.get(COLORMAP_SETTING) is None

    def test_the_preference_is_the_operators_and_survives_the_dialog(
        self, session: SessionViewModel, tmp_path: Path
    ) -> None:
        """It follows the person, not the work (ADR-0047) — so it is on disk in
        `settings.json` and not in any project directory."""
        dialog = SettingsDialog(session)
        dialog.colormap.setCurrentText("cividis")

        dialog.apply()

        assert "cividis" in (tmp_path / "settings.json").read_text()


class TestAProjectThatOverrides:
    """M4-T10's sentence, with its first caller."""

    def test_it_says_so_rather_than_showing_a_value_the_edit_will_not_change(
        self, app: Nanoscope, tmp_path: Path
    ) -> None:
        root = tmp_path / "P"
        SqliteProjectRepository.create(root, "P").close()
        session = SessionViewModel(app)
        session.open_project(root)
        app.settings.set(COLORMAP_SETTING, "bone", Scope.PROJECT)

        dialog = SettingsDialog(session)

        assert session.overridden_by_project(COLORMAP_SETTING)
        assert _notes(dialog) == ["This project overrides your default; the project's value wins."]

    def test_nothing_is_said_when_nothing_overrides(self, session: SessionViewModel) -> None:
        dialog = SettingsDialog(session)

        assert not _notes(dialog)


class TestTheViewerFollowsTheDefault:
    def test_a_changed_default_reaches_an_open_viewer(self, session: SessionViewModel) -> None:
        """The combo is *this scan*; the dialog is *the default*. One reads the
        key and the other writes it, so they cannot fight over it."""
        viewer = ImageViewer(session)
        assert viewer.colormap.currentText() == COLORMAPS[0]

        session.remember(COLORMAP_SETTING, "viridis")

        assert viewer.colormap.currentText() == "viridis"

    def test_a_new_viewer_opens_in_the_stored_default(self, session: SessionViewModel) -> None:
        session.remember(COLORMAP_SETTING, "bone")

        assert ImageViewer(session).colormap.currentText() == "bone"


def _notes(dialog: SettingsDialog) -> list[str]:
    from PySide6.QtWidgets import QLabel

    return [
        label.text()
        for label in dialog.findChildren(QLabel)
        if "overrides your default" in label.text()
    ]


class TestWhoseValueIsShown:
    def test_the_control_shows_the_operators_value_not_the_projects(
        self, app: Nanoscope, tmp_path: Path
    ) -> None:
        """It edits the operator's scope, so showing the merged value would put
        a project's answer in a control that writes somewhere else — and OK
        would then copy that project's choice into every other project, which is
        ADR-0047's first failure mode exactly."""
        root = tmp_path / "P"
        SqliteProjectRepository.create(root, "P").close()
        session = SessionViewModel(app)
        session.open_project(root)
        session.remember(COLORMAP_SETTING, "magma")
        app.settings.set(COLORMAP_SETTING, "bone", Scope.PROJECT)

        dialog = SettingsDialog(session)

        assert dialog.colormap.currentText() == "magma"
        assert session.preference(COLORMAP_SETTING) == "bone"
        assert _notes(dialog)
