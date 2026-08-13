"""The main window: menus, docks, a status bar, and no business logic (M5-T02).

The first widget in the project, and the one every rule so far was written to
protect. Architecture §2.3: *"a widget may format, lay out, and emit signals. It
may not decide **what** to compute, **which** device to use, or **how** to
measure."*

So this class **holds the container and constructs nothing**. Opening a project
is `Nanoscope.open`, which is the composition root's job (M5-T01); what this file
adds is a menu item, a status message, and somewhere to put the answer.

The docks are placeholders that name the task filling them. A dock with a label
saying "M5-T04" is honest; a dock with half a project explorer in it is M5-T04
started early and finished badly.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from nanoscope.core.entities.project import OpenedProject
from nanoscope.gui.dialogs import ImportOptions
from nanoscope.gui.panels import ImageViewer, JobStatus, ProjectExplorer, PropertiesPanel
from nanoscope.gui.viewmodels import SessionViewModel

if TYPE_CHECKING:
    from nanoscope.app.container import Nanoscope

logger = logging.getLogger(__name__)

#: Application-scope settings keys. A window layout follows the **operator**, not
#: the work, so it goes beside their colormap and not into a project directory
#: (ADR-0047's rule for choosing a scope).
GEOMETRY_SETTING = "window.geometry"
STATE_SETTING = "window.state"

#: Each dock, and the task that fills it. Written as data so the next task
#: replaces one line instead of hunting for its placeholder.
DOCKS: tuple[tuple[str, str, Qt.DockWidgetArea], ...] = (
    ("Log", "The log panel arrives in M5-T08.", Qt.DockWidgetArea.BottomDockWidgetArea),
)

#: The docks with a panel in them (M5-T04, M5-T06). The rest are placeholders,
#: and each names the task that replaces it.
PROJECT_DOCK = "Project"
PROPERTIES_DOCK = "Properties"


class MainWindow(QMainWindow):
    """One window, holding the application it displays."""

    def __init__(self, app: Nanoscope) -> None:
        super().__init__()
        self._app = app
        #: The one viewmodel, constructed here because this is the one place a
        #: panel is constructed (ADR-0057). Panels subscribe to it; nothing
        #: subscribes to a panel.
        self._last_failure = ""
        self.session = SessionViewModel(app, self)
        self.session.project_changed.connect(self._project_changed)
        self.session.image_changed.connect(self._image_changed)
        self.session.failed.connect(self._failed)
        self.session.reported.connect(self.statusBar().showMessage)
        self.session.job_changed.connect(self._job_changed)

        self.setWindowTitle("nanoscope")
        self.viewer = ImageViewer(self.session, self)
        self.viewer.readout.connect(self.statusBar().showMessage)
        self.setCentralWidget(self.viewer)
        self._build_docks()
        self._build_menus()
        #: Permanent, so a job's progress is not wiped by the next status
        #: message — and on the right, where it does not fight the readout.
        self.jobs = JobStatus(self.session, self)
        self.statusBar().addPermanentWidget(self.jobs)
        self.statusBar().showMessage("No project open")

        self._restore_layout()

    # ── Building ──────────────────────────────────────────────────────────────

    def _build_docks(self) -> None:
        self.explorer = ProjectExplorer(self.session, self)
        project_dock = QDockWidget(PROJECT_DOCK, self)
        project_dock.setObjectName("dock.project")
        project_dock.setWidget(self.explorer)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, project_dock)

        self.properties = PropertiesPanel(self.session, self)
        properties_dock = QDockWidget(PROPERTIES_DOCK, self)
        properties_dock.setObjectName("dock.properties")
        properties_dock.setWidget(self.properties)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, properties_dock)

        for title, message, area in DOCKS:
            dock = QDockWidget(title, self)
            #: Named so `saveState()` can find it again — a dock without an
            #: object name is one the restored layout silently drops.
            dock.setObjectName(f"dock.{title.lower()}")
            dock.setWidget(_placeholder(message))
            self.addDockWidget(area, dock)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("toolbar.main")

        self.open_action = QAction("&Open Project…", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.choose_project)

        self.import_action = QAction("&Import Images…", self)
        self.import_action.setShortcut("Ctrl+I")
        self.import_action.triggered.connect(self.choose_images)
        self.import_action.setEnabled(False)

        self.remove_action = QAction("&Remove Image from Project", self)
        self.remove_action.triggered.connect(self.explorer.remove_selected)
        self.remove_action.setEnabled(False)

        self.close_action = QAction("&Close Project", self)
        self.close_action.triggered.connect(self.close_project)
        self.close_action.setEnabled(False)

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        for action in (self.open_action, self.import_action, self.close_action):
            file_menu.addAction(action)
            toolbar.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.remove_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        for dock in self.findChildren(QDockWidget):
            view_menu.addAction(dock.toggleViewAction())

    # ── What the actions do ───────────────────────────────────────────────────

    def choose_project(self) -> None:
        """Ask for a directory, then open it. The dialog is the only thing here."""
        directory = QFileDialog.getExistingDirectory(self, "Open Project")
        if directory:
            self.open_project(directory)

    def choose_images(self) -> None:
        """Ask which files, then how to read them, then run it as a job.

        Two dialogs because they are two questions, and the second one is worth
        asking once for the batch: a folder of scans comes off one instrument
        (ADR-0041's argument for one `modality` per call).
        """
        files, _ = QFileDialog.getOpenFileNames(self, "Import Images")
        if not files:
            return

        options = ImportOptions(self)
        if options.exec() != ImportOptions.DialogCode.Accepted:
            return

        choice = options.choice()
        self.session.import_images(
            files,
            modality=choice.modality,
            pixel_size_nm=choice.pixel_size_nm,
        )

    def open_project(self, project_dir: Path | str) -> OpenedProject | None:
        """Open a project through the session, and say what happened.

        Separate from `choose_project` so a test — and, later, a "recent
        projects" menu — can open one without a modal dialog.

        Returns:
            What was opened, or `None` if it was refused.
        """
        opened = self.session.open_project(project_dir)
        if opened is None:
            #: A dialog here and nowhere else: the operator pressed a button
            #: labelled with this action. A failure to *display* a scan is a
            #: status line, because they clicked a row (ADR-0056).
            QMessageBox.warning(self, "Cannot open that", self._last_failure)
        return opened

    def close_project(self) -> None:
        self.session.close_project()

    # ── What the session says back ────────────────────────────────────────────

    def _project_changed(self, opened: OpenedProject | None) -> None:
        """The window's own chrome. The panels heard the same signal."""
        self.setWindowTitle("nanoscope" if opened is None else f"{opened.name} — nanoscope")
        self.statusBar().showMessage("No project open" if opened is None else _summarise(opened))
        self._update_actions()

    def _image_changed(self, _image: object) -> None:
        self._update_actions()

    def _job_changed(self, _job: object) -> None:
        self._update_actions()

    def _update_actions(self) -> None:
        """One place decides what can be pressed, because three signals change it.

        **While a job runs, the project may not be taken out from under it:**
        `close_project()` closes the SQLite connection the worker thread is
        using, and opening another replaces it. Cancelling stays available — it
        is the button this task exists to make honest.

        **Removal follows the selection, not the load.** A scan whose file is
        missing loads as `None` and is still selected, and forgetting it is the
        likeliest thing an operator wants to do next.
        """
        busy = self.session.is_busy
        has_project = self.session.project is not None
        self.open_action.setEnabled(not busy)
        self.close_action.setEnabled(not busy and has_project)
        self.import_action.setEnabled(not busy and has_project)
        self.remove_action.setEnabled(not busy and self.session.image_id is not None)

    def _failed(self, message: str) -> None:
        """Our errors are messages (ADR-0030, ADR-0052 §3). A traceback in a
        dialog is an application blaming its user for its diagnostics. The
        viewmodel logged it; what is added here is showing it."""
        self._last_failure = message
        self.statusBar().showMessage(message)

    # ── Layout, remembered ────────────────────────────────────────────────────

    def _restore_layout(self) -> None:
        """Put the window back where the operator left it, if they left it anywhere.

        A layout from an older version can be unreadable — Qt says so by
        returning `False` — and the answer is the default layout, not a refusal
        to start.
        """
        for key, restore in (
            (GEOMETRY_SETTING, self.restoreGeometry),
            (STATE_SETTING, self.restoreState),
        ):
            stored = self._app.settings.get(key)
            if isinstance(stored, str) and stored and not restore(_decode(stored)):
                logger.info("ignoring an unreadable stored %s", key)

    def save_layout(self) -> None:
        """Remember where things are, in the operator's settings and not the project's.

        Base64 because Qt hands back bytes and the store is JSON; the
        application scope because a window layout follows the person, not the
        work (ADR-0047).
        """
        self._app.settings.set(GEOMETRY_SETTING, _encode(self.saveGeometry()))
        self._app.settings.set(STATE_SETTING, _encode(self.saveState()))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt's name
        self.save_layout()
        super().closeEvent(event)


def _summarise(opened: OpenedProject) -> str:
    """One line for the status bar, including what is wrong.

    The integrity report is *shown* here too — ADR-0040's obligation is not
    discharged once and forgotten; every surface that opens a project owes it.
    """
    line = f"{opened.name}: {len(opened.images)} image(s)"
    report = opened.integrity
    if report.is_clean:
        return line
    return (
        f"{line} — {len(report.missing_files)} missing file(s), "
        f"{len(report.untracked_files)} untracked"
    )


def _placeholder(message: str) -> QWidget:
    label = QLabel(message)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    return label


def _encode(data: QByteArray) -> str:
    """Qt's bytes as text, because the settings store is JSON (ADR-0047 §3).

    `.data()` rather than `bytes(...)`: PySide6's stubs do not describe the
    `QByteArray` → `bytes` conversion, and a `type: ignore` for something the
    API offers directly is an ignore that outlives its reason.
    """
    return base64.b64encode(data.data()).decode("ascii")


def _decode(text: str) -> QByteArray:
    try:
        return QByteArray(base64.b64decode(text.encode("ascii")))
    except (ValueError, UnicodeEncodeError):
        return QByteArray()
