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
from nanoscope.core.errors import NanoscopeError
from nanoscope.gui.panels import ProjectExplorer

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
    (
        "Properties",
        "Image and run properties arrive in M5-T06.",
        Qt.DockWidgetArea.RightDockWidgetArea,
    ),
    ("Log", "The log panel arrives in M5-T08.", Qt.DockWidgetArea.BottomDockWidgetArea),
)

#: The one dock with a panel in it (M5-T04). The others are still placeholders,
#: and each names the task that replaces it.
PROJECT_DOCK = "Project"


class MainWindow(QMainWindow):
    """One window, holding the application it displays."""

    def __init__(self, app: Nanoscope) -> None:
        super().__init__()
        self._app = app

        self.setWindowTitle("nanoscope")
        self.setCentralWidget(_placeholder("The image viewer arrives in M5-T05."))
        self._build_docks()
        self._build_menus()
        self.statusBar().showMessage("No project open")

        self._restore_layout()

    # ── Building ──────────────────────────────────────────────────────────────

    def _build_docks(self) -> None:
        self.explorer = ProjectExplorer(self._app, self)
        project_dock = QDockWidget(PROJECT_DOCK, self)
        project_dock.setObjectName("dock.project")
        project_dock.setWidget(self.explorer)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, project_dock)

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

        self.remove_action = QAction("&Remove Image from Project", self)
        self.remove_action.triggered.connect(self.explorer.remove_selected)
        self.remove_action.setEnabled(False)
        self.explorer.image_selected.connect(lambda _: self.remove_action.setEnabled(True))

        self.close_action = QAction("&Close Project", self)
        self.close_action.triggered.connect(self.close_project)
        self.close_action.setEnabled(False)

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        for action in (self.open_action, self.close_action):
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

    def open_project(self, project_dir: Path | str) -> OpenedProject | None:
        """Open a project through the container, and say what happened.

        Separate from `choose_project` so a test — and, later, a "recent
        projects" menu — can open one without a modal dialog.

        Returns:
            What was opened, or `None` if it was refused.
        """
        try:
            opened = self._app.open(project_dir)
        except NanoscopeError as refusal:
            #: Our errors are messages (ADR-0030, ADR-0052 §3). A traceback in a
            #: dialog is an application blaming its user for its diagnostics.
            self._refuse(str(refusal))
            return None

        self.explorer.show_project(opened)
        self.setWindowTitle(f"{opened.name} — nanoscope")
        self.close_action.setEnabled(True)
        self.statusBar().showMessage(_summarise(opened))
        return opened

    def close_project(self) -> None:
        self._app.close_project()
        self.explorer.show_project(None)
        self.remove_action.setEnabled(False)
        self.setWindowTitle("nanoscope")
        self.close_action.setEnabled(False)
        self.statusBar().showMessage("No project open")

    def _refuse(self, message: str) -> None:
        logger.error("refused to open a project: %s", message)
        self.statusBar().showMessage(message)
        QMessageBox.warning(self, "Cannot open that", message)

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
