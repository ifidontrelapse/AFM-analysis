"""Turning what was drawn into a model, from the window (M8-T05, ADR-0085).

Four tasks built a training module nothing called. M8-T01 wrote the port,
M8-T02 the dataset, M8-T03 the trainer and M8-T04 the record, and each one ended
on the same line: *not wired into the composition root; the caller arrives with
M8-T05.* This is that caller.

**Modeless, and not a dock.** Modeless because M5's third exit criterion rules
out the alternative in as many words — *a long-running job shows progress and can
be cancelled without freezing the UI* — and a modal window over a six-hour run is
the frozen application it is reporting on. Closing it does not stop the run: the
run is in the project (ADR-0084) and the status bar already shows the job without
knowing what training is, which is what M8-T03 routed progress through
`JobContext.report` for.

Not a dock because the nine that exist answer *what about this scan?* — and
`apply_default_layout` carries the measurement that made the right-hand group
tabbed: 811 px of minimum height against 1 785 for the same five untabbed. A
tenth panel would compete for that space to ask a question that is not about the
selected image.

**No model name is written here.** `TrainingConfig.base_model` is required and
`gui/` is grepped for those words (PROJECT_RULES §2.5, D-19's lesson), so the
starting points come from `application` and this file renders what it is handed.
The metric columns come from `METRIC_BLOCKS` for the same reason one layer on:
ADR-0080 declared the vocabulary once and predicted it would grow, and a widget
with its own column list is the copy that drifts.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.use_cases import StartingPoint
from nanoscope.core.entities.training import METRIC_BLOCKS, TrainingConfig, TrainingRun
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: Where an option is kept on its combo entry — the same role the detection
#: panel uses, for the same reason: the label is for a person and the value is
#: for the application.
_OPTION = Qt.ItemDataRole.UserRole

#: What "no preference" is, in a list of devices. `None` reaches the provider as
#: *let the manager decide*, which is the answer that keeps working when the
#: operator changes machines (ADR-0049).
AUTOMATIC = "Automatic (best available)"

#: The two scopes an operator can train on, and the first is the default because
#: of what the second means. ADR-0044: *a model trained on its own output is
#: confirming itself* — so including adopted boxes is a choice that gets made
#: out loud, which is M7-T09's reading of the same rule one milestone earlier.
SCOPES: tuple[tuple[str, bool], ...] = (
    ("Hand-drawn boxes only", True),
    ("Every annotation, adopted ones included", False),
)

#: The epoch table's columns, derived from the vocabulary rather than typed.
#: `epoch` first because it is the row's identity; the rest in the order
#: `METRIC_BLOCKS` declares them, so a new block appears here the day it is
#: added to `core` and not one release later.
METRICS: tuple[str, ...] = tuple(name for names in METRIC_BLOCKS.values() for name in names)

#: What a stored run is called when its id belongs to a process that is gone.
#: **Not "failed"** — nobody observed a failure, and inventing one is the
#: substitution ADR-0025 removed for scales and ADR-0033 for heights (ADR-0084 §8).
INTERRUPTED = "interrupted"


class TrainingDialog(QDialog):
    """Configure a run, watch it, stop it, and read what this project has run."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("Train a Model")
        #: Modeless: the reason is in the module docstring, and it is the whole
        #: shape of this window.
        self.setModal(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self._configuration())
        layout.addLayout(self._buttons())
        layout.addWidget(self.status)
        layout.addWidget(self._progress_group())
        layout.addWidget(self._history_group())

        #: Connected after the widgets exist rather than where each is made:
        #: a handler that reads a button built two methods later is a crash on
        #: the first keystroke, and nothing about the order says so.
        self.model_id.textChanged.connect(lambda _text: self._update_buttons())
        session.training_changed.connect(self.show_run)
        session.job_changed.connect(lambda _job: self._update_buttons())
        session.project_changed.connect(lambda _project: self.reload())
        self.reload()

    # ── What it is made of ────────────────────────────────────────────────────

    def _configuration(self) -> QGroupBox:
        self.model_id = QLineEdit(self)
        self.model_id.setPlaceholderText("what to call the model this produces")
        self.model_id.setToolTip(
            "The name a detection configuration will use. An operator names their\n"
            "model; registering under a name that already exists replaces it,\n"
            "which is what retraining means."
        )

        self.start_from = QComboBox(self)
        self.scope = QComboBox(self)
        for label, hand_drawn_only in SCOPES:
            self.scope.addItem(label, hand_drawn_only)

        self.hold_out = QDoubleSpinBox(self)
        self.hold_out.setRange(0.0, 0.9)
        self.hold_out.setSingleStep(0.05)
        self.hold_out.setValue(0.2)
        self.hold_out.setToolTip(
            "How much to hold back, by scan and never by box.\n"
            "0 trains on everything and reports no validation numbers at all,\n"
            "because there would be nothing honest to compute them on."
        )

        self.epochs = QSpinBox(self)
        self.epochs.setRange(1, 10_000)
        self.epochs.setValue(100)

        self.image_size = QSpinBox(self)
        self.image_size.setRange(32, 4096)
        self.image_size.setSingleStep(32)
        self.image_size.setValue(640)

        #: Zero is *no answer*, not a batch of none. `TrainingConfig` says why
        #: the field is `None`-able: a batch size is a decision about memory
        #: this layer cannot see, and the framework can.
        self.batch_size = QSpinBox(self)
        self.batch_size.setRange(0, 1024)
        self.batch_size.setSpecialValueText("Let the framework decide")
        self.batch_size.setValue(0)

        self.device = QComboBox(self)
        self.device.addItem(AUTOMATIC, None)
        for device in self._session.devices():
            #: The devices this machine has, not the four the enum names — three
            #: of which would fail on any given computer (ADR-0049).
            self.device.addItem(f"{device.name} ({device.kind})", device.kind)

        self.seed = QSpinBox(self)
        self.seed.setRange(0, 1_000_000)
        self.seed.setToolTip(
            "Which shuffle decides the split. Recorded in the dataset, because\n"
            "two runs split differently cannot be compared."
        )

        form = QFormLayout()
        form.addRow("Call the model:", self.model_id)
        form.addRow("Start from:", self.start_from)
        form.addRow("Train on:", self.scope)
        form.addRow("Hold out:", self.hold_out)
        form.addRow("Epochs:", self.epochs)
        form.addRow("Image size (px):", self.image_size)
        form.addRow("Batch size:", self.batch_size)
        form.addRow("Device:", self.device)
        form.addRow("Seed:", self.seed)

        box = QGroupBox("Configuration", self)
        box.setLayout(form)
        return box

    def _buttons(self) -> QHBoxLayout:
        self.train = QPushButton("Train", self)
        self.train.clicked.connect(self.start)
        self.stop = QPushButton("Stop", self)
        self.stop.setEnabled(False)
        self.stop.setToolTip(
            "Asks the run to stop at its next epoch boundary.\n"
            "What was trained by then is kept; nothing is registered."
        )
        self.stop.clicked.connect(self._stop_pressed)

        self.status = QLabel("No run yet.", self)
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        row = QHBoxLayout()
        row.addWidget(self.train)
        row.addWidget(self.stop)
        row.addStretch(1)
        return row

    def _progress_group(self) -> QGroupBox:
        self.epoch_table = QTableWidget(0, 1 + len(METRICS), self)
        self.epoch_table.setHorizontalHeaderLabels(["Epoch", *METRICS])
        self.epoch_table.verticalHeader().setVisible(False)
        self.epoch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.epoch_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        box = QGroupBox("This run, epoch by epoch", self)
        inner = QVBoxLayout(box)
        inner.addWidget(self.epoch_table)
        return box

    def _history_group(self) -> QGroupBox:
        self.history = QTableWidget(0, 5, self)
        self.history.setHorizontalHeaderLabels(
            ["Started", "Status", "Epochs", "Held out", "Weights"]
        )
        self.history.verticalHeader().setVisible(False)
        self.history.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        box = QGroupBox("What this project has run", self)
        inner = QVBoxLayout(box)
        inner.addWidget(self.history)
        return box

    # ── What it shows ─────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Rebuild everything that depends on which project is open."""
        self.start_from.clear()
        for point in self._session.starting_points():
            self.start_from.addItem(point.label, point)
            self.start_from.setItemData(
                self.start_from.count() - 1, point.detail, Qt.ItemDataRole.ToolTipRole
            )
        self.show_run(self._session.training)
        self.reload_history()

    def reload_history(self) -> None:
        """What the **project** recorded, newest first (ADR-0084 §1).

        A `running` row whose id no live provider knows is a run interrupted by
        a process that died mid-epoch. It is shown as that and never as failed:
        nobody observed a failure, and there is no `resume` to offer either
        (ADR-0080's named negative).
        """
        runs = self._session.training_runs()
        self.history.setRowCount(len(runs))
        for row, run in enumerate(runs):
            interrupted = not run.is_finished and not self._session.is_live(run)
            cells = (
                run.started_utc,
                INTERRUPTED if interrupted else str(run.status),
                f"{run.epochs_done} of {run.config.epochs}",
                str(run.dataset.val_images),
                run.weights_path or "",
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if interrupted:
                    item.setToolTip(
                        "This process is not running it. The record says what was "
                        "true when it stopped being written to; nothing here can "
                        "honestly say whether it finished."
                    )
                self.history.setItem(row, column, item)

    def show_run(self, run: TrainingRun | None) -> None:
        """Describe the live run, on the **main** thread (ADR-0058 §1).

        The signal that carries it is queued, which is what makes touching
        widgets here safe: the port publishes from the provider's thread.
        """
        self._update_buttons()
        if run is None:
            self.status.setText("No run yet.")
            self.epoch_table.setRowCount(0)
            return

        self.status.setText(_describe(run))
        #: **A block nothing measured is not a column of blanks.** ADR-0082 made
        #: `validation` mean *a held-out set existed*, so a run with nothing held
        #: out has no precision to show and no honest place to show one — and a
        #: `precision` header over five empty cells is a question an operator
        #: spends the run wondering about. Hidden, not filled with zeros.
        reported = {name for one in run.metrics for name in one.values}
        for column, name in enumerate(METRICS, start=1):
            self.epoch_table.setColumnHidden(column, name not in reported)

        self.epoch_table.setRowCount(len(run.metrics))
        for row, epoch in enumerate(run.metrics):
            self.epoch_table.setItem(row, 0, QTableWidgetItem(str(epoch.epoch)))
            for column, name in enumerate(METRICS, start=1):
                value = epoch.values.get(name)
                #: Empty rather than zero for a metric this epoch did not carry:
                #: a 0.000 is a score, and the absence of one is not.
                text = "" if value is None else f"{value:.4f}"
                self.epoch_table.setItem(row, column, QTableWidgetItem(text))

        if run.is_finished:
            self.reload_history()

    def _update_buttons(self) -> None:
        training = self._session.is_training
        self.train.setEnabled(
            bool(self.model_id.text().strip())
            and self._session.project is not None
            and not training
            and not self._session.is_busy
        )
        self.stop.setEnabled(training)

    # ── What it does ──────────────────────────────────────────────────────────

    def config(self) -> TrainingConfig | None:
        """What the form is asking for, or `None` when it may not ask."""
        point: StartingPoint | None = self.start_from.currentData(_OPTION)
        if point is None or not self.model_id.text().strip():
            return None
        return TrainingConfig(
            base_model=point.base_model,
            epochs=self.epochs.value(),
            image_size_px=self.image_size.value(),
            #: Zero is the special value above: *no answer*, which the entity
            #: spells `None` and refuses as a batch size.
            batch_size=self.batch_size.value() or None,
            device=self.device.currentData(_OPTION),
            seed=self.seed.value(),
        )

    def start(self) -> None:
        config = self.config()
        if config is None:
            return
        self._session.train(
            config,
            model_id=self.model_id.text().strip(),
            hand_drawn_only=bool(self.scope.currentData(_OPTION)),
            val_fraction=self.hold_out.value(),
            seed=self.seed.value(),
        )
        self.epoch_table.setRowCount(0)
        self.status.setText("Building the dataset…")
        self._update_buttons()

    def _stop_pressed(self) -> None:
        self.stop.setEnabled(False)
        #: *Stopping*, not stopped — the same word `JobStatus` uses, for the same
        #: reason: the request is recorded and the work stops where it can, which
        #: for a run is the next epoch boundary (ADR-0043 §3).
        self.status.setText("Stopping at the next epoch boundary…")
        self._session.cancel_training()


def _describe(run: TrainingRun) -> str:
    """One line about a run, in the order the questions are asked."""
    where = f" on {run.device.name}" if run.device is not None else ""
    line = f"{run.status}{where}: epoch {run.epochs_done} of {run.config.epochs}"
    if run.error:
        return f"{line} — {run.error}"
    if run.weights_path:
        return f"{line}; weights at {run.weights_path}"
    return line
