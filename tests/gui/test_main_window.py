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
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMenu,
    QMessageBox,
    QToolBar,
)

from nanoscope.app.container import Nanoscope
from nanoscope.core.values import Modality
from nanoscope.gui.main_window import (
    ANNOTATE_DOCK,
    DETECTION_DOCK,
    GEOMETRY_SETTING,
    LOG_DOCK,
    MEASUREMENTS_DOCK,
    PREPROCESSING_DOCK,
    PROFILE_DOCK,
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
            ANNOTATE_DOCK,
            PROFILE_DOCK,
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
            "AnnotatePanel",
            "ProfilePanel",
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
        means is that the second window matches the first.

        A **third** clamp arrived on 2026-08-30 and it is the reason for the skip:
        a stored layout that does not fit the screen is deliberately not restored
        (`_reject_a_layout_that_does_not_fit`), and whether this window fits the
        offscreen platform's 800x800 depends on whether an earlier test has
        applied the theme. Skipped rather than made conditional, because an
        assertion that cannot fail is worse than one that does not run — and the
        decision it would be reaching for is stated with explicit sizes in
        `test_a_layout_that_fits_is_left_alone`.
        """
        first = MainWindow(app)
        available = first.screen().availableGeometry().size()  # type: ignore[union-attr]
        if first.minimumSizeHint().height() > available.height():
            pytest.skip("this window does not fit this screen, so the layout guard owns the case")
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
        assert len(window.findChildren(QDockWidget)) == 9

    def test_a_first_run_has_no_geometry_to_restore(self, app: Nanoscope) -> None:
        """The flag the launcher reads to decide whether to maximise."""
        assert not MainWindow(app).restored_geometry

    def test_a_dock_layout_too_tall_for_the_screen_is_not_restored(self, app: Nanoscope) -> None:
        """The defect that made the application unusable, found 2026-08-30.

        `restoreState` puts the docks back exactly as they were, **including
        untabbed**, and the five right-hand panels side by side vertically ask
        for more minimum height than a screen has. A window is never smaller
        than its layout's minimum, so no size and no maximise helps: the bottom
        of it — the status bar, a running job's progress, three docks — sits
        below the edge of the monitor and cannot be clicked.

        Measured on the machine that found it: a minimum of **883x1785** against
        a **2048x1152** screen, where the layout this application ships asks for
        **883x811**. Those are the numbers the test states, because the
        offscreen platform's own screen is smaller than this window's minimum
        and could not stage the other outcome.
        """
        window = MainWindow(app)
        for dock in window._right:
            window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        untabbed = window.minimumSizeHint().height()
        assert not window.tabifiedDockWidgets(window._right[0]), "the premise: they are apart"

        window._reject_a_layout_that_does_not_fit(QSize(2048, 1152))

        assert window.minimumSizeHint().height() < untabbed
        assert window.minimumSizeHint().height() <= 1152
        assert window.tabifiedDockWidgets(window._right[0]), (
            "the fallback is the layout this application ships, which is the tabbed one"
        )

    def test_the_geometry_goes_with_it(self, app: Nanoscope) -> None:
        """The second half, and the reason the first is not enough.

        Qt's `restoreGeometry` clamps a stored *size* to the available screen on
        its own — that part is not this project's to assert. What it does not
        clamp is the **minimum**, so a window carrying the layout above is
        oversized whatever geometry was stored, and putting the docks back is
        not finished until the window is a size that fits too.
        """
        window = MainWindow(app)
        window.restored_geometry = True
        window.resize(3000, 3000)

        window._reject_a_layout_that_does_not_fit(QSize(2048, 1152))

        assert window.width() <= 2048
        assert window.height() <= max(1152, window.minimumSizeHint().height())
        assert not window.restored_geometry, "so the launcher maximises it"

    def test_a_layout_that_fits_is_left_alone(self, app: Nanoscope) -> None:
        """The guard is not "always start over": a layout and a size that fit
        the screen come back untouched. A layout is a preference (ADR-0047)."""
        window = MainWindow(app)
        window.restored_geometry = True
        window.resize(900, 900)
        tabbed = window.minimumSizeHint()

        window._reject_a_layout_that_does_not_fit(QSize(2048, 1152))

        assert window.restored_geometry
        assert (window.width(), window.height()) == (900, 900)
        assert window.minimumSizeHint() == tabbed

    def test_an_unreadable_geometry_is_not_a_restored_one(self, app: Nanoscope) -> None:
        """Unreadable is not "stored": the fallback is the default layout, which
        the launcher then maximises."""
        app.application_settings.set_setting(GEOMETRY_SETTING, "not base64 at all!!")

        assert not MainWindow(app).restored_geometry

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


class TestAProjectCanBeMade:
    """The gap an operator hits on an empty machine.

    `SqliteProjectRepository.create` has existed since M4-T04 and was called
    only by tests, while `Import Images…` stays disabled until a project is
    open — so there was no way in: no project could be made, and therefore no
    image could be imported.
    """

    def test_new_project_makes_one_and_opens_it(
        self, app: Nanoscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = MainWindow(app)
        root = tmp_path / "Gold on mica"
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(root))

        window.new_action.trigger()

        assert app.repository is not None
        assert app.repository.name == "Gold on mica"
        assert (root / "images").is_dir()
        assert (root / "project.json").is_file()

    def test_the_new_project_can_take_images(
        self, app: Nanoscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The point of the action: the import that was disabled is now live."""
        window = MainWindow(app)
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(tmp_path / "New")
        )

        window.new_action.trigger()

        assert window.import_action.isEnabled()

    def test_a_directory_with_files_in_it_is_refused_with_a_sentence(
        self, app: Nanoscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writing a manifest into somebody's folder turns it into a project.

        The refusal is the repository's and reaches the status bar unchanged —
        a second copy of the rule in the widget is the copy that goes stale.
        """
        occupied = tmp_path / "Photos"
        occupied.mkdir()
        (occupied / "holiday.jpg").write_bytes(b"not a scan")
        window = MainWindow(app)
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(occupied)
        )

        window.new_action.trigger()

        assert app.repository is None
        assert "not empty" in window.statusBar().currentMessage()

    def test_a_cancelled_dialog_makes_nothing(
        self, app: Nanoscope, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = MainWindow(app)
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: "")

        window.new_action.trigger()

        assert app.repository is None
