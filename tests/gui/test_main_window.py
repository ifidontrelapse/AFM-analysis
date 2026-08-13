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
from PySide6.QtWidgets import QApplication, QDockWidget, QMenu, QMessageBox, QToolBar

from nanoscope.app.container import Nanoscope
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import (
    DETECTION_DOCK,
    GEOMETRY_SETTING,
    LOG_DOCK,
    MEASUREMENTS_DOCK,
    PREPROCESSING_DOCK,
    PROJECT_DOCK,
    PROPERTIES_DOCK,
    STATE_SETTING,
    STATISTICS_DOCK,
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

        assert docks == {
            PROJECT_DOCK,
            PROPERTIES_DOCK,
            LOG_DOCK,
            PREPROCESSING_DOCK,
            DETECTION_DOCK,
            MEASUREMENTS_DOCK,
            STATISTICS_DOCK,
        }

    def test_every_dock_has_a_panel_in_it(self, app: Nanoscope) -> None:
        """M5-T02 filled each dock with a label naming the task that would
        replace it. **M5-T08 replaced the last one**, so the assertion changes
        from "every placeholder names its task" to "there are no placeholders" —
        a promise is only worth making while it is outstanding."""
        window = MainWindow(app)

        widgets = [dock.widget() for dock in window.findChildren(QDockWidget)]

        assert {type(widget).__name__ for widget in widgets} == {
            "ProjectExplorer",
            "PropertiesPanel",
            "LogPanel",
            "PreprocessingPanel",
            "DetectionPanel",
            "MeasurementsPanel",
            "StatisticsPanel",
        }

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
        """Compared against what the first window **actually got**, not against
        what it was asked for. Two clamps sit in between: the offscreen platform
        has an 800x600 virtual screen, and the docks' minimum sizes grow as
        panels arrive — M6-T05's table pushed the minimum width past 640, which
        turned an absolute assertion into an order-dependent one. What the test
        means is that the second window matches the first."""
        first = MainWindow(app)
        first.resize(640, 480)
        saved = (first.width(), first.height())
        first.save_layout()

        second = MainWindow(app)

        assert (second.width(), second.height()) == saved

    def test_an_unreadable_layout_is_ignored_rather_than_fatal(self, app: Nanoscope) -> None:
        """A layout from an older version can be unreadable; the answer is the
        default layout, not a refusal to start."""
        app.application_settings.set_setting(GEOMETRY_SETTING, "not base64 at all!!")
        app.application_settings.set_setting(STATE_SETTING, "@@@")

        window = MainWindow(app)

        assert window.statusBar().currentMessage() == "No project open"
        assert len(window.findChildren(QDockWidget)) == 7

    def test_closing_the_window_saves_the_layout(self, app: Nanoscope) -> None:
        window = MainWindow(app)
        window.resize(800, 600)

        window.close()

        assert app.application_settings.get_setting(GEOMETRY_SETTING)


class TestTheWindowIsWiredThroughTheSession:
    """M5-T06: the window connects panels to the viewmodel and to nothing else.

    What is asserted here is the wiring, since that is what changed: a selection
    reaches the viewer without the explorer knowing the viewer exists, and a
    refusal to *display* is a status line while a refusal to *open* is a dialog.
    """

    def test_a_selection_travels_through_the_viewmodel(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)

        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

        assert window.session.image_id == window.explorer.selected_image_id

    def test_removal_follows_the_selection_and_not_the_load(
        self, app: Nanoscope, project: Path
    ) -> None:
        """The fixture's scan is three bytes of `b"AFM"`, so it does not load —
        and forgetting it is exactly what an operator would want to do next."""
        window = MainWindow(app)
        window.open_project(project)

        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

        assert window.session.image is None
        assert window.remove_action.isEnabled()

    def test_a_display_refusal_is_a_status_line_and_not_a_dialog(
        self, app: Nanoscope, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-0056, kept: the operator clicked a row, not a button labelled
        "load". The dialog belongs to the action they asked for by name."""
        shown: list[object] = []
        said: list[str] = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: shown.append(args))
        window = MainWindow(app)
        window.open_project(project)
        window.session.failed.connect(said.append)

        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

        assert shown == []
        assert window.statusBar().currentMessage() == said[0]

    def test_closing_a_project_empties_every_panel(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        window.open_project(project)
        window.explorer.tree.setCurrentItem(window.explorer.tree.topLevelItem(0))

        window.close_project()

        assert window.explorer.tree.topLevelItemCount() == 0
        assert window.properties.values["Name"].text() == "—"
        assert not window.remove_action.isEnabled()


class TestWhileAJobRuns:
    """M5-T07: what may be pressed while something is running in the background.

    `close_project()` closes the SQLite connection the worker thread is using,
    and opening another replaces it — so those actions are off until it ends.
    Cancelling is not, because that is the button the task exists to make honest.
    """

    def test_the_actions_that_would_pull_the_project_away_are_disabled(
        self, app: Nanoscope, project: Path, tmp_path: Path
    ) -> None:
        import threading

        import numpy as np

        window = MainWindow(app)
        window.open_project(project)
        source = tmp_path / "extra.npy"
        np.save(source, np.zeros((8, 8), dtype=np.float32))

        repository = app.repository
        assert repository is not None
        original = repository.import_image
        in_flight, release = threading.Event(), threading.Event()

        def slow(*args: object, **kwargs: object) -> object:
            in_flight.set()
            release.wait(5.0)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        repository.import_image = slow  # type: ignore[method-assign]
        job = window.session.import_images([source], modality=Modality.AFM)
        assert job is not None
        assert in_flight.wait(5.0)
        QApplication.processEvents()

        assert not window.open_action.isEnabled()
        assert not window.close_action.isEnabled()
        assert not window.import_action.isEnabled()
        #: `isHidden`, not `isVisible`: a child of a window that was never
        #: shown is not visible whatever the strip decided about itself.
        assert not window.jobs.isHidden()

        release.set()
        assert job.wait(5.0)
        QApplication.processEvents()

        assert window.open_action.isEnabled()
        assert window.import_action.isEnabled()
        assert window.jobs.isHidden()

    def test_importing_needs_a_project(self, app: Nanoscope, project: Path) -> None:
        window = MainWindow(app)
        assert not window.import_action.isEnabled()

        window.open_project(project)

        assert window.import_action.isEnabled()
