"""The main window: menus, docks, a status bar, and no business logic (M5-T02).

The first widget in the project, and the one every rule so far was written to
protect. Architecture §2.3: *"a widget may format, lay out, and emit signals. It
may not decide **what** to compute, **which** device to use, or **how** to
measure."*

So this class **holds the container and constructs nothing**. Opening a project
is `Nanoscope.open`, which is the composition root's job (M5-T01); what this file
adds is a menu item, a status message, and somewhere to put the answer.

M5-T02 filled the docks with labels naming the task that would replace each —
*"a dock with a label saying M5-T04 is honest; a dock with half a project
explorer in it is M5-T04 started early and finished badly"*. **M5-T08 replaced
the last of them**, so what is left is four panels and the wiring between them.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox

from nanoscope.app.logging import attach_view_log, detach_view_log
from nanoscope.core.entities.project import OpenedProject
from nanoscope.gui.dialogs import ImportOptions, LabelSource, SettingsDialog
from nanoscope.gui.panels import (
    AnnotatePanel,
    DetectionPanel,
    ImageViewer,
    JobStatus,
    LogPanel,
    MeasurementsPanel,
    PreprocessingPanel,
    ProfilePanel,
    ProjectExplorer,
    PropertiesPanel,
    StatisticsPanel,
)
from nanoscope.gui.viewmodels import SessionViewModel
from nanoscope.gui.viewmodels.log_stream import LogLine, LogStream

if TYPE_CHECKING:
    from nanoscope.app.container import Nanoscope

logger = logging.getLogger(__name__)

#: Application-scope settings keys. A window layout follows the **operator**, not
#: the work, so it goes beside their colormap and not into a project directory
#: (ADR-0047's rule for choosing a scope).
GEOMETRY_SETTING = "window.geometry"
STATE_SETTING = "window.state"

#: The docks, and the task that filled each. M5-T02 wrote them as placeholders
#: naming the task that would replace them; **M5-T08 was the last of those**, so
#: there is no placeholder left and no list of promises to keep.
PROJECT_DOCK = "Project"  # M5-T04
PROPERTIES_DOCK = "Properties"  # M5-T06
LOG_DOCK = "Log"  # M5-T08
PREPROCESSING_DOCK = "Preprocessing"  # M6-T01
DETECTION_DOCK = "Detection"  # M6-T02
MEASUREMENTS_DOCK = "Measurements"  # M6-T05
STATISTICS_DOCK = "Statistics"  # M6-T06
ANNOTATE_DOCK = "Annotate"  # M7-T02
PROFILE_DOCK = "Profile"  # M7-T06


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
        self.session.run_changed.connect(self._run_changed)
        #: **The history says when it moved**, and the Undo menu is labelled by
        #: what it would take back. M7-T02 read that off `annotations_changed`
        #: while every command mutated annotations and wrote down that the first
        #: one that did not would need a signal; M7-T05's ruler was it, and a
        #: second layer signal was added beside the first. A third would have
        #: been the same mistake again (M7-T08, ADR-0077).
        self.session.history_changed.connect(self._update_actions)

        #: The log reaches the screen through one handler, attached by `app/`
        #: because that is the only layer allowed to attach one (ADR-0051), and
        #: detached in `closeEvent` because a handler pointing at a deleted
        #: widget turns the next log line into a crash.
        self.log_stream = LogStream(self)
        self.log_stream.logged.connect(self._logged)
        self._unseen = 0
        attach_view_log(self.log_stream.handler)

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
        #: Where in the project the operator is. Permanent, because half of
        #: navigating is knowing whether there is anywhere left to go (M6-T08).
        self.position = QLabel("", self)
        self.statusBar().addPermanentWidget(self.position)
        self.statusBar().showMessage("No project open")
        #: Every action starts in the state the *session* implies, rather than
        #: in whatever Qt's default is: `next_action` was enabled with no
        #: project open until a test asked (M6-T08).
        self._update_actions()

        #: Whether this window came back to a size the operator chose. Read by
        #: the launcher, which maximises when it did not (`gui/launcher.py`).
        self.restored_geometry = False
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

        self.preprocessing = PreprocessingPanel(self.session, self)
        preprocessing_dock = QDockWidget(PREPROCESSING_DOCK, self)
        preprocessing_dock.setObjectName("dock.preprocessing")
        preprocessing_dock.setWidget(self.preprocessing)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, preprocessing_dock)
        #: Tabbed with Properties rather than stacked: they answer different
        #: questions about the same scan, and both want the height.
        self.tabifyDockWidget(properties_dock, preprocessing_dock)
        properties_dock.raise_()

        self.detection = DetectionPanel(self.session, self)
        detection_dock = QDockWidget(DETECTION_DOCK, self)
        detection_dock.setObjectName("dock.detection")
        detection_dock.setWidget(self.detection)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, detection_dock)
        self.tabifyDockWidget(preprocessing_dock, detection_dock)
        properties_dock.raise_()

        self.measurements = MeasurementsPanel(self.session, self)
        measurements_dock = QDockWidget(MEASUREMENTS_DOCK, self)
        measurements_dock.setObjectName("dock.measurements")
        measurements_dock.setWidget(self.measurements)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, measurements_dock)

        self.statistics = StatisticsPanel(self.session, self)
        statistics_dock = QDockWidget(STATISTICS_DOCK, self)
        statistics_dock.setObjectName("dock.statistics")
        statistics_dock.setWidget(self.statistics)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, statistics_dock)
        self.tabifyDockWidget(detection_dock, statistics_dock)

        self.annotate = AnnotatePanel(self.session, self)
        annotate_dock = QDockWidget(ANNOTATE_DOCK, self)
        annotate_dock.setObjectName("dock.annotate")
        annotate_dock.setWidget(self.annotate)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, annotate_dock)
        self.tabifyDockWidget(statistics_dock, annotate_dock)
        properties_dock.raise_()
        #: The tool drives the canvas, and the canvas hands back what was drawn —
        #: both through the window that owns them, never panel to panel
        #: (ADR-0057).
        self.annotate.draw.toggled.connect(self.viewer.view.set_drawing)
        self.viewer.view.box_drawn.connect(self.annotate.box_drawn)
        self.annotate.outline.toggled.connect(self.viewer.view.set_outlining)
        self.viewer.view.polygon_drawn.connect(self.annotate.polygon_drawn)
        self.annotate.brush.toggled.connect(self.viewer.view.set_painting)
        self.annotate.brush_size.valueChanged.connect(self.viewer.view.set_brush)
        self.viewer.view.mask_painted.connect(self.annotate.mask_painted)
        self.annotate.measure.toggled.connect(self.viewer.view.set_measuring)
        self.viewer.view.line_drawn.connect(self.annotate.line_drawn)

        self.profile = ProfilePanel(self.session, self)
        profile_dock = QDockWidget(PROFILE_DOCK, self)
        profile_dock.setObjectName("dock.profile")
        profile_dock.setWidget(self.profile)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, profile_dock)
        self.tabifyDockWidget(measurements_dock, profile_dock)

        self.log = LogPanel(self.log_stream, self)
        self.log_dock = QDockWidget(LOG_DOCK, self)
        self.log_dock.setObjectName("dock.log")
        self.log_dock.setWidget(self.log)
        #: Looking at it is what marks it read, so the count resets on the
        #: signal Qt already emits for that.
        self.log_dock.visibilityChanged.connect(self._log_visibility_changed)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        #: Tabbed with the log: both are things one reads *after* a run, and the
        #: bottom of the window has room for one of them at a time.
        self.tabifyDockWidget(measurements_dock, self.log_dock)
        measurements_dock.raise_()

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

        self.next_action = QAction("&Next Image", self)
        self.next_action.setShortcut("Ctrl+Right")
        self.next_action.triggered.connect(self.session.select_next)
        self.previous_action = QAction("&Previous Image", self)
        self.previous_action.setShortcut("Ctrl+Left")
        self.previous_action.triggered.connect(self.session.select_previous)

        self.export_run_action = QAction("&Export This Run…", self)
        self.export_run_action.triggered.connect(lambda: self.export(everything=False))
        self.export_run_action.setEnabled(False)

        self.export_all_action = QAction("Export &All Measurements…", self)
        self.export_all_action.triggered.connect(lambda: self.export(everything=True))
        self.export_all_action.setEnabled(False)

        #: Two items rather than one with a hidden default, ADR-0067's rule one
        #: milestone on — and here the scope is what keeps a training set honest:
        #: boxes adopted from a detector are the model's own output (ADR-0044).
        self.export_hand_drawn_action = QAction("Export &Hand-Drawn Annotations…", self)
        self.export_hand_drawn_action.setToolTip(
            "Only the boxes a person drew. A model trained on boxes copied from "
            "its own output is confirming itself."
        )
        self.export_hand_drawn_action.triggered.connect(
            lambda: self.export_annotations(hand_drawn_only=True)
        )
        self.export_hand_drawn_action.setEnabled(False)

        self.export_annotations_action = QAction("Export All A&nnotations…", self)
        self.export_annotations_action.setToolTip(
            "Every annotation, including the ones adopted from a detector."
        )
        self.export_annotations_action.triggered.connect(
            lambda: self.export_annotations(hand_drawn_only=False)
        )
        self.export_annotations_action.setEnabled(False)

        self.import_annotations_action = QAction("Import Anno&tations…", self)
        self.import_annotations_action.triggered.connect(self.choose_labels)
        self.import_annotations_action.setEnabled(False)

        edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self._undo)
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        self.redo_action.triggered.connect(self._redo)
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        self.settings_action = QAction("Se&ttings…", self)
        self.settings_action.triggered.connect(self.edit_settings)

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        for action in (self.open_action, self.import_action, self.close_action):
            file_menu.addAction(action)
            toolbar.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.remove_action)
        file_menu.addSeparator()
        #: Two scopes, named rather than implied: a single item that silently
        #: meant one of them is one somebody uses wrong once (ADR-0067).
        file_menu.addAction(self.export_run_action)
        file_menu.addAction(self.export_all_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_hand_drawn_action)
        file_menu.addAction(self.export_annotations_action)
        file_menu.addAction(self.import_annotations_action)
        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.previous_action)
        view_menu.addAction(self.next_action)
        view_menu.addSeparator()
        for action in (self.previous_action, self.next_action):
            toolbar.addAction(action)
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

    def edit_settings(self) -> None:
        """Open the preferences. The dialog stores and applies; this opens it."""
        SettingsDialog(self.session, self).exec()

    def _undo(self) -> None:
        """The session steps the history and says so; nothing to refresh here."""
        self.session.undo()

    def _redo(self) -> None:
        self.session.redo()

    def export(self, *, everything: bool) -> None:
        """Ask for a CSV. The session writes it; the status bar says where."""
        self.session.export(everything=everything)
        self._update_actions()

    def export_annotations(self, *, hand_drawn_only: bool) -> None:
        """Write the labels. Which scope, the menu item already said (M7-T09)."""
        self.session.export_annotations(hand_drawn_only=hand_drawn_only)
        self._update_actions()

    def choose_labels(self) -> None:
        """Ask which directory, then the one question the files cannot answer.

        Two dialogs because they are two questions, and the second one is the
        provenance M8 depends on — asked once for the batch, like the modality of
        an image import (ADR-0041, M5-T07).
        """
        directory = QFileDialog.getExistingDirectory(self, "Import Annotations")
        if not directory:
            return

        dialog = LabelSource(self)
        if dialog.exec() != LabelSource.DialogCode.Accepted:
            return
        self.session.import_annotations(directory, source=dialog.choice())

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

    def _run_changed(self, _run: object) -> None:
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
        self.export_run_action.setEnabled(not busy and self.session.run is not None)
        self.export_all_action.setEnabled(not busy and has_project)
        self.export_hand_drawn_action.setEnabled(not busy and has_project)
        self.export_annotations_action.setEnabled(not busy and has_project)
        self.import_annotations_action.setEnabled(not busy and has_project)

        #: Labelled by *what they would take back*: "Undo" alone makes an
        #: operator press it to find out (M4-T08 wrote the labels for this).
        undo, redo = self.session.undo_label, self.session.redo_label
        self.undo_action.setEnabled(not busy and undo is not None)
        self.undo_action.setText("&Undo" if undo is None else f"&Undo {undo}")
        self.redo_action.setEnabled(not busy and redo is not None)
        self.redo_action.setText("&Redo" if redo is None else f"&Redo {redo}")

        position = self.session.position_text()
        self.position.setText(position)
        where = self.session.image_position
        self.previous_action.setEnabled(not busy and where is not None and where[0] > 1)
        self.next_action.setEnabled(not busy and where is not None and where[0] < where[1])

    def _logged(self, line: LogLine) -> None:
        """Count what an operator has not looked at, and say so in the title.

        Not a toast and not an auto-raised panel: the two ways desktop
        notifications fail are being missed and being resented, and a count in a
        dock's title is the version that is neither. `INFO` does not notify —
        a notification for every ordinary line is the same as none (ADR-0059).
        """
        if not line.is_notable or self.log_dock.isVisible():
            return
        self._unseen += 1
        self.log_dock.setWindowTitle(f"{LOG_DOCK} ({self._unseen})")

    def _log_visibility_changed(self, visible: bool) -> None:
        if visible:
            self._unseen = 0
            self.log_dock.setWindowTitle(LOG_DOCK)

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

        Sets `restored_geometry`, which the launcher reads: with nothing stored
        Qt sizes the window from its `sizeHint`, and this one is a viewer
        surrounded by nine docks — the hint is a window too small to work in.
        **The decision to maximise instead is the launcher's**, because it is a
        decision about *showing* the window, and a constructor that shows itself
        is one a test cannot build offscreen without one appearing.
        """
        for key, restore in (
            (GEOMETRY_SETTING, self.restoreGeometry),
            (STATE_SETTING, self.restoreState),
        ):
            stored = self._app.settings.get(key)
            if isinstance(stored, str) and stored:
                if restore(_decode(stored)):
                    self.restored_geometry = self.restored_geometry or key == GEOMETRY_SETTING
                else:
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
        detach_view_log()
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
