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

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox

from nanoscope.app.logging import attach_view_log, detach_view_log
from nanoscope.core.entities.project import OpenedProject
from nanoscope.gui.dialogs import (
    ImageChooser,
    ImportOptions,
    LabelSource,
    ModelsDialog,
    SettingsDialog,
    TrainingDialog,
)
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
        #: A run starting and a run ending change what may be pressed, and
        #: neither is a `Job` this window would otherwise hear about — ADR-0080
        #: §2 kept the two apart on purpose (M8-T05).
        self.session.training_changed.connect(self._training_changed)

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

        #: Kept rather than made per press: it is modeless, so a second press
        #: while a run is going must raise the window that is watching it and
        #: not open a second one beside it (M8-T05).
        self.training_dialog: TrainingDialog | None = None

    # ── Building ──────────────────────────────────────────────────────────────

    def _build_docks(self) -> None:
        self.explorer = ProjectExplorer(self.session, self)
        project_dock = QDockWidget(PROJECT_DOCK, self)
        project_dock.setObjectName("dock.project")
        project_dock.setWidget(self.explorer)

        self.properties = PropertiesPanel(self.session, self)
        properties_dock = QDockWidget(PROPERTIES_DOCK, self)
        properties_dock.setObjectName("dock.properties")
        properties_dock.setWidget(self.properties)

        self.preprocessing = PreprocessingPanel(self.session, self)
        preprocessing_dock = QDockWidget(PREPROCESSING_DOCK, self)
        preprocessing_dock.setObjectName("dock.preprocessing")
        preprocessing_dock.setWidget(self.preprocessing)

        self.detection = DetectionPanel(self.session, self)
        detection_dock = QDockWidget(DETECTION_DOCK, self)
        detection_dock.setObjectName("dock.detection")
        detection_dock.setWidget(self.detection)

        self.measurements = MeasurementsPanel(self.session, self)
        measurements_dock = QDockWidget(MEASUREMENTS_DOCK, self)
        measurements_dock.setObjectName("dock.measurements")
        measurements_dock.setWidget(self.measurements)

        self.statistics = StatisticsPanel(self.session, self)
        statistics_dock = QDockWidget(STATISTICS_DOCK, self)
        statistics_dock.setObjectName("dock.statistics")
        statistics_dock.setWidget(self.statistics)

        self.annotate = AnnotatePanel(self.session, self)
        annotate_dock = QDockWidget(ANNOTATE_DOCK, self)
        annotate_dock.setObjectName("dock.annotate")
        annotate_dock.setWidget(self.annotate)
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

        self.log = LogPanel(self.log_stream, self)
        self.log_dock = QDockWidget(LOG_DOCK, self)
        self.log_dock.setObjectName("dock.log")
        self.log_dock.setWidget(self.log)
        #: Looking at it is what marks it read, so the count resets on the
        #: signal Qt already emits for that.
        self.log_dock.visibilityChanged.connect(self._log_visibility_changed)

        #: Where each dock goes when nobody has moved it, kept so the layout can
        #: be applied a **second** time — `_restore_layout` falls back to it when
        #: a stored one cannot fit the screen.
        self._right = (
            properties_dock,
            preprocessing_dock,
            detection_dock,
            statistics_dock,
            annotate_dock,
        )
        self._bottom = (measurements_dock, profile_dock, self.log_dock)
        self._left = (project_dock,)
        self.apply_default_layout()

    def apply_default_layout(self) -> None:
        """Put every dock back where the application puts it.

        Applied when the window is built, and **again** when a stored layout
        turns out not to fit the screen — which is why it is a method rather
        than a run of `addDockWidget` calls inline in `_build_docks`.

        **The right-hand panels are tabbed, not stacked**, and that is the whole
        of why the default fits: they answer different questions about the same
        scan and each wants the height, so five of them side by side vertically
        ask for 1 464 px before the canvas has any. Tabbed, the group asks for
        the tallest one. Measured on the machine that found this: 811 px of
        minimum height against 1 785 for the same five untabbed.
        """
        for docks, area in (
            (self._left, Qt.DockWidgetArea.LeftDockWidgetArea),
            (self._right, Qt.DockWidgetArea.RightDockWidgetArea),
            (self._bottom, Qt.DockWidgetArea.BottomDockWidgetArea),
        ):
            for dock in docks:
                #: A restored layout may have left it floating or hidden, and
                #: `addDockWidget` alone does not undo either.
                dock.setFloating(False)
                dock.setVisible(True)
                self.addDockWidget(area, dock)

        for group in (self._right, self._bottom):
            for dock in group[1:]:
                self.tabifyDockWidget(group[0], dock)

        #: Properties on top of the right-hand stack: it describes the scan that
        #: was just selected, which is the question asked first.
        self._right[0].raise_()
        #: Measurements on top of the bottom one, for the same reason one task
        #: later — it is what a finished run produces.
        self._bottom[0].raise_()

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("toolbar.main")

        self.new_action = QAction("&New Project…", self)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.triggered.connect(self.create_project)

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

        self.models_action = QAction("&Models…", self)
        self.models_action.setToolTip("What this project can detect with, and which model it uses.")
        self.models_action.triggered.connect(self.manage_models)
        self.models_action.setEnabled(False)

        self.train_action = QAction("Train a &Model…", self)
        self.train_action.setToolTip(
            "Build a dataset from this project's annotations and train on it."
        )
        self.train_action.triggered.connect(self.open_training)
        self.train_action.setEnabled(False)

        self.settings_action = QAction("Se&ttings…", self)
        self.settings_action.triggered.connect(self.edit_settings)

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        for action in (self.new_action, self.open_action, self.import_action, self.close_action):
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
        file_menu.addAction(self.train_action)
        file_menu.addAction(self.models_action)
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

    def create_project(self) -> None:
        """Ask where to put a new project, then make it and open it.

        One dialog, not two. Qt's directory chooser already offers *New Folder*,
        and the project's display name defaults to the directory's — an operator
        who names a folder has named the project, and a second dialog asking
        them to say it again is the one they close without reading. The name and
        the directory are allowed to differ (M4-T04); nothing here forces them to.

        A directory with files in it is **refused by the repository**, not
        checked for here: the refusal is one sentence, it already exists, and a
        second copy of the rule in a widget is the copy that goes stale.
        """
        directory = QFileDialog.getExistingDirectory(self, "New Project")
        if directory:
            self.session.create_project(directory, Path(directory).name)

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

        The first is `ImageChooser` rather than `getOpenFileNames`: the static
        helper returns a native dialog with nowhere to put a preview, and a scan
        is chosen by what is in it, not by an acquisition number in its name.
        """
        chooser = ImageChooser(self)
        if chooser.exec() != ImageChooser.DialogCode.Accepted:
            return

        files = chooser.chosen()
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

    def open_training(self) -> None:
        """Show the training window, or raise the one already open (M8-T05).

        `show()` and not `exec()`: it is the one modeless dialog in this
        application, because M5's third exit criterion says a long job is
        watched *without freezing the UI*, and a modal window over six hours of
        training is that freeze with a progress bar on it.
        """
        if self.training_dialog is None:
            self.training_dialog = TrainingDialog(self.session, self)
        self.training_dialog.show()
        self.training_dialog.raise_()
        self.training_dialog.activateWindow()

    def manage_models(self) -> None:
        """What this project can detect with, and which model it uses (M8-T06).

        Modal, unlike the training window: this asks a question and takes an
        answer, where that one watches a run (ADR-0085 §1, ADR-0086).
        """
        ModelsDialog(self.session, self).exec()

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

    def _training_changed(self, _run: object) -> None:
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
        #: **Two questions, not one.** `is_busy` is one short job owning the
        #: project's connection; `is_training` is a run that lasts hours. Only
        #: the three that would pull the project out from under the trainer read
        #: both — an operator can annotate, undo and export while a model
        #: trains, because the repository's own lock already serialises the two
        #: writers (M8-T05, and `_serialised` says so).
        occupied = busy or self.session.is_training
        self.new_action.setEnabled(not occupied)
        self.open_action.setEnabled(not occupied)
        self.close_action.setEnabled(not occupied and has_project)
        self.import_action.setEnabled(not busy and has_project)
        self.remove_action.setEnabled(not busy and self.session.image_id is not None)
        self.export_run_action.setEnabled(not busy and self.session.run is not None)
        self.export_all_action.setEnabled(not busy and has_project)
        self.export_hand_drawn_action.setEnabled(not busy and has_project)
        self.export_annotations_action.setEnabled(not busy and has_project)
        self.import_annotations_action.setEnabled(not busy and has_project)
        #: Opening the window is not starting a run, so this stays available
        #: while one is going — it is where the Stop button lives.
        self.train_action.setEnabled(has_project)
        self.models_action.setEnabled(has_project)

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

        **A stored layout that does not fit the screen is not restored.** See
        `_reject_a_layout_that_does_not_fit`, which runs after both halves are
        back, because it is the restored dock state that decides the minimum.
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

        self._reject_a_layout_that_does_not_fit()

    def _reject_a_layout_that_does_not_fit(self, available: QSize | None = None) -> None:
        """A stored layout that cannot be reached with a mouse is not restored.

        Two stored values, two ways to be unreachable, one rule. Both were live
        on an operator's machine on 2026-08-30, and the second is the one that
        made the application unusable:

        **The dock layout.** `restoreState` puts the docks back exactly as they
        were, including *untabbed*, and five right-hand panels side by side
        vertically ask for 1 464 px of minimum height before the canvas has any.
        Measured there: a **minimum** of 883x1785 against a 2048x1152 screen. A
        window cannot be smaller than its layout's minimum, so it does not
        matter what size it is given or whether it is maximised — the bottom of
        it, with the status bar, the progress of a running job and three docks
        in it, is below the edge of the monitor and cannot be clicked. The
        remedy is the layout this application ships, which asks for 811.

        **The geometry.** `restoreGeometry` restores a size saved on whatever
        display was in use last, and Qt clamps the *position* onto a screen
        rather than shrinking the *size* to one. That machine's `settings.json`
        held 1025x1797, written when the monitor reported 4096x2304 at a device
        pixel ratio of 1 and read back after it began reporting half that at a
        ratio of 2. Nothing was corrupt; Qt restored what it was given.

        Height and width both, and size only: on Wayland the compositor owns
        placement and a client's idea of its own position is not something to
        decide from, while a window that is too *big* is unreachable everywhere.

        Neither is a corruption, so neither is an error — `save_layout` writes
        the working layout back on close, and the next launch restores that.

        Args:
            available: the room there is, for a test that needs to state it.
                Defaults to the screen's, which is what the application uses —
                and the reason the parameter exists is that the offscreen test
                platform's screen is smaller than this window's own minimum, so
                a test that took its numbers from there could only ever stage
                one of the two outcomes.
        """
        if available is None:
            screen = self.screen()
            if screen is None:  # pragma: no cover — a built window always has one
                return
            available = screen.availableGeometry().size()

        needed = self.minimumSizeHint()
        if needed.height() > available.height() or needed.width() > available.width():
            logger.info(
                "the stored dock layout needs %dx%d and the screen is %dx%d; "
                "using the default layout instead",
                needed.width(),
                needed.height(),
                available.width(),
                available.height(),
            )
            self.apply_default_layout()

        if self.width() > available.width() or self.height() > available.height():
            logger.info(
                "the stored geometry is %dx%d and the screen is %dx%d; ignoring it",
                self.width(),
                self.height(),
                available.width(),
                available.height(),
            )
            #: Resized as well as un-restored, so that *un*-maximising later
            #: gives a window that fits rather than putting the operator back
            #: into the state this method exists to leave. Qt clamps it up to
            #: the window's own minimum, which is the honest floor.
            self.resize(available)
            self.restored_geometry = False

    def save_layout(self) -> None:
        """Remember where things are, in the operator's settings and not the project's.

        Base64 because Qt hands back bytes and the store is JSON; the
        application scope because a window layout follows the person, not the
        work (ADR-0047).
        """
        self._app.settings.set(GEOMETRY_SETTING, _encode(self.saveGeometry()))
        self._app.settings.set(STATE_SETTING, _encode(self.saveState()))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt's name
        """Ask before walking away from a run, then close.

        **Measured, and this is why it asks.** `Nanoscope.close()` calls
        `jobs.shutdown(wait=True)`, so a six-second job made `close()` take
        6.01 s and was never asked to stop — with a training run that is hours
        of a process with no window, no progress and no cancel button. And
        `wait=False` fixes nothing: it returned in 0.00 s and the process still
        took the full 5.06 s to exit, because `concurrent.futures` joins its
        threads at interpreter exit.

        So the honest thing is to ask, and to cancel on the way out — which
        lands at the next epoch boundary, which is all ADR-0043 ever promised.
        """
        if self.session.is_training and not self._confirm_abandoning_the_run():
            event.ignore()
            return
        #: Asked, not awaited. What was trained by the boundary is kept, and
        #: nothing is registered — a cancelled run has no weights to register
        #: (ADR-0084 §5).
        self.session.cancel_training()
        self.save_layout()
        detach_view_log()
        super().closeEvent(event)

    def _confirm_abandoning_the_run(self) -> bool:
        """Whether to close anyway. The question, and what it costs to say yes."""
        answer = QMessageBox.question(
            self,
            "A model is still training",
            "Closing stops the run at its next epoch boundary, so the window may "
            "stay a little longer.\n\nWhat was trained by then is kept on disk and "
            "recorded in the project, but no model is registered.\n\nClose anyway?",
            QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Close


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
