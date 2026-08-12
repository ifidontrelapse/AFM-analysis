"""The window, headless (M5-T02, ADR-0053).

M5's fourth exit criterion is *"GUI smoke tests pass headless in CI"*, and this
is the first file that can fail it. What is asserted is deliberately narrow —
that the window **holds** the application rather than doing its work:

- opening a project goes through the container and the title says so;
- a refusal is a message in the status bar, not a traceback;
- the layout is saved into the *operator's* settings and restored from them;
- every dock names the task that will fill it, so an empty panel is a promise
  rather than a bug.

Nothing here checks pixels. A widget that renders correctly and calls the wrong
use case is the failure this project's rules are written against.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QDockWidget, QMenu, QMessageBox, QToolBar

from nanoscope.app.container import Nanoscope
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import (
    DOCKS,
    GEOMETRY_SETTING,
    PROJECT_DOCK,
    STATE_SETTING,
    MainWindow,
)
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "Gold on mica"
    with SqliteProjectRepository.create(root, "Gold on mica") as repo:
        (repo.root / "images" / "scan.spm").write_bytes(b"AFM")
        repo.add_image("images/scan.spm", modality=Modality.AFM, pixel_size_nm=1.95)
    return root


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        yield container


class TestTheWindowIsBuilt:
    def test_it_has_the_menus_the_shell_promises(self, app: Nanoscope) -> None:
        window = MainWindow(app)

        titles = [menu.title() for menu in window.menuBar().findChildren(QMenu)]

        assert "&File" in titles
        assert "&View" in titles

    def test_it_has_a_toolbar_and_a_status_bar(self, app: Nanoscope) -> None:
        window = MainWindow(app)

        assert [bar.objectName() for bar in window.findChildren(QToolBar)] == ["toolbar.main"]
        assert window.statusBar().currentMessage() == "No project open"

    def test_every_dock_is_there(self, app: Nanoscope) -> None:
        window = MainWindow(app)

        docks = {dock.windowTitle() for dock in window.findChildren(QDockWidget)}

        assert docks == {PROJECT_DOCK, *(title for title, _, _ in DOCKS)}

    def test_the_docks_without_a_panel_yet_say_which_task_fills_them(self, app: Nanoscope) -> None:
        """An empty panel is a promise when it names its task, and a bug when it
        does not. The Project dock is no longer on this list — M5-T04 filled
        it, which is what the promise was for."""
        window = MainWindow(app)

        docks = {dock.windowTitle(): dock for dock in window.findChildren(QDockWidget)}

        for title, message, _ in DOCKS:
            assert "M5-T" in message
            assert message in docks[title].widget().text()

    def test_docks_are_named_so_a_saved_layout_can_find_them(self, app: Nanoscope) -> None:
        """A dock without an object name is one the restored layout silently
        drops — Qt warns about it and then forgets the panel."""
        window = MainWindow(app)

        assert all(dock.objectName() for dock in window.findChildren(QDockWidget))

    def test_the_close_action_is_off_until_something_is_open(self, app: Nanoscope) -> None:
        window = MainWindow(app)

        assert window.open_action.isEnabled()
        assert not window.close_action.isEnabled()


class TestOpeningAProject:
    def test_it_goes_through_the_container(self, app: Nanoscope, project: Path) -> None:
        """The window holds the application; it does not open projects itself."""
        window = MainWindow(app)

        opened = window.open_project(project)

        assert opened is not None
        assert app.repository is not None
        assert window.windowTitle() == "Gold on mica — nanoscope"
        assert window.close_action.isEnabled()

    def test_the_status_bar_summarises_it(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)

        window.open_project(project)

        assert window.statusBar().currentMessage() == "Gold on mica: 1 image(s)"

    def test_the_status_bar_carries_the_integrity_report(
        self, app: Nanoscope, project: Path
    ) -> None:
        """ADR-0040's obligation is not discharged once and forgotten: every
        surface that opens a project owes it."""
        (project / "images" / "scan.spm").unlink()
        window = MainWindow(app)

        window.open_project(project)

        assert "1 missing file(s)" in window.statusBar().currentMessage()

    def test_a_refusal_is_a_message_not_a_traceback(
        self, app: Nanoscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dialog is suppressed here because a modal box blocks a headless
        run; what is asserted is that the refusal was *caught* and shown."""
        monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
        window = MainWindow(app)

        assert window.open_project(tmp_path / "not-a-project") is None
        assert "not a project directory" in window.statusBar().currentMessage()

    def test_closing_a_project_puts_the_window_back(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)

        window.close_project()

        assert window.windowTitle() == "nanoscope"
        assert not window.close_action.isEnabled()
        assert app.repository is None


class TestTheLayoutIsRemembered:
    def test_saving_writes_into_the_operators_settings(
        self, app: Nanoscope, tmp_path: Path
    ) -> None:
        """A window layout follows the person, not the work — so it lands in
        `settings.json` and never in a project directory (ADR-0047)."""
        window = MainWindow(app)

        window.save_layout()

        assert isinstance(app.application_settings.get_setting(GEOMETRY_SETTING), str)
        assert isinstance(app.application_settings.get_setting(STATE_SETTING), str)
        assert (tmp_path / "settings.json").is_file()

    def test_a_new_window_restores_what_the_last_one_saved(self, app: Nanoscope) -> None:
        """The size is small on purpose: the offscreen platform has an 800x600
        virtual screen and clamps a window to it, so asking for 1234 wide gets
        798 back and the test fails for a reason that has nothing to do with
        the code under it."""
        first = MainWindow(app)
        first.resize(640, 480)
        first.save_layout()

        second = MainWindow(app)

        assert (second.width(), second.height()) == (640, 480)

    def test_an_unreadable_layout_is_ignored_rather_than_fatal(self, app: Nanoscope) -> None:
        """A layout from an older version can be unreadable; the answer is the
        default layout, not a refusal to start."""
        app.application_settings.set_setting(GEOMETRY_SETTING, "not base64 at all!!")
        app.application_settings.set_setting(STATE_SETTING, "@@@")

        window = MainWindow(app)

        assert window.statusBar().currentMessage() == "No project open"
        assert len(window.findChildren(QDockWidget)) == len(DOCKS) + 1  # + the Project dock

    def test_closing_the_window_saves_the_layout(self, app: Nanoscope) -> None:
        window = MainWindow(app)
        window.resize(800, 600)

        window.close()

        assert app.application_settings.get_setting(GEOMETRY_SETTING)
