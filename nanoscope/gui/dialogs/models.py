"""The models this project has, and which one it detects with (M8-T06, ADR-0086).

M8's third exit criterion is written against this dialog:

> *A trained model is selectable for detection in M6 **with no code change**.*

Until now it was measurably false, and the reason was **W10** — a line M4-T13
named, made closable, and handed to M5, which did not pay it. The weights path
on `PipelineConfig` defaulted to a checkpoint under `./checkpoints/`, resolved
against whatever directory the process started in: `True` from the repository
root, where an untracked file sits, and `False` from anywhere else. So M8-T05
could produce a model and nothing could select it.

**Four verbs, and the fourth is not what it sounds like.** *Import* registers
weights that already exist; *register* is what training does; *activate* writes
the project-scoped setting a detection run reads. **Compare is the records** —
what each model was trained on, on how many images, its classes, its input size,
when it was registered, and whether the file is still there. Comparing models by
*running* them is M8-T08's evaluation report through the M3-T15 harness, and a
second answer to that question invented here would be the copy that drifts.

Modal, unlike the training window: this asks a question and takes an answer,
where that one watches six hours of work (ADR-0085 §1).

**No model or framework name is written here.** The frameworks come from the
registry through the session, and PROJECT_RULES §2.5 is checked by a grep.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nanoscope.core.entities.model import ModelDescriptor, ModelTask
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: Where a value is kept on a row or a combo entry.
_VALUE = Qt.ItemDataRole.UserRole

#: What the table says about a model, in the order the questions are asked: what
#: is it called, is it the one in use, what does it do, how big is its input,
#: what can it name, where did it come from, and **is the file still there**.
COLUMNS = (
    "Model",
    "Active",
    "Task",
    "Framework",
    "Input (px)",
    "Classes",
    "Registered",
    "Weights",
    "Provenance",
)

#: What a row says when the weights the row points at are not on this machine.
#: ADR-0040's dangling row, from the model side — shown rather than hidden,
#: because hiding it turns *"that model is elsewhere"* into *"that model never
#: existed"*, and it is the reason an activation is refused.
MISSING = "missing"

#: The mark on the row a detection run will load. A word rather than a colour:
#: a table whose one important fact is a shade is a table nobody can screenshot
#: into a bug report.
ACTIVE = "in use"


class ModelsDialog(QDialog):
    """List what is registered, register more, and choose the one that runs."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("Models")

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._update_buttons)

        self.activate = QPushButton("Use for Detection", self)
        self.activate.clicked.connect(self._activate_selected)
        self.detect_with_nothing = QPushButton("Use None", self)
        self.detect_with_nothing.setToolTip(
            "Detect with no model until one is chosen.\n"
            "A detector that needs weights then refuses before it reads a file."
        )
        self.detect_with_nothing.clicked.connect(lambda: self._activate(None))

        self.note = QLabel("", self)
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        buttons = QHBoxLayout()
        buttons.addWidget(self.activate)
        buttons.addWidget(self.detect_with_nothing)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(self.note)
        layout.addWidget(self._import_group())

        session.settings_changed.connect(self.reload)
        session.project_changed.connect(lambda _project: self.reload())
        self.reload()

    # ── Registering weights that already exist ────────────────────────────────

    def _import_group(self) -> QGroupBox:
        self.weights = QLineEdit(self)
        self.weights.setPlaceholderText("a weights file already on this machine")
        self.weights.textChanged.connect(lambda _text: self._update_buttons())

        self.browse = QPushButton("Browse…", self)
        self.browse.clicked.connect(self.choose_weights)

        self.new_id = QLineEdit(self)
        self.new_id.setPlaceholderText("what this project will call it")
        self.new_id.textChanged.connect(lambda _text: self._update_buttons())

        #: The three things a `.pt` file does not say, asked rather than
        #: guessed — `ImportOptions`' shape since M5-T07, `LabelSource`'s since
        #: M7-T09, and this is the third time the same answer is right.
        self.task = QComboBox(self)
        for task in ModelTask:
            self.task.addItem(str(task), task)

        self.framework = QComboBox(self)
        for framework in self._session.frameworks():
            self.framework.addItem(str(framework), framework)

        self.provenance = QLineEdit(self)
        self.provenance.setPlaceholderText("where it came from, in your own words")
        self.provenance.setToolTip(
            "Free text on purpose: provenance that has to fit a schema stops\n"
            "being recorded (ADR-0005). A model this project trained fills it in."
        )

        self.register = QPushButton("Register", self)
        self.register.clicked.connect(self._register)

        chooser = QHBoxLayout()
        chooser.addWidget(self.weights)
        chooser.addWidget(self.browse)

        form = QFormLayout()
        form.addRow("Weights:", chooser)
        form.addRow("Call it:", self.new_id)
        form.addRow("It does:", self.task)
        form.addRow("Loaded by:", self.framework)
        form.addRow("Came from:", self.provenance)
        form.addRow("", self.register)

        box = QGroupBox("Register weights from this machine", self)
        box.setLayout(form)
        box.setToolTip(
            "The file is registered where it is and never copied: a 137 MB\n"
            "checkpoint is not duplicated into every project that uses it, and\n"
            "the consequence is that this project opens elsewhere without it."
        )
        return box

    def choose_weights(self) -> None:
        chosen, _filter = QFileDialog.getOpenFileName(self, "Choose weights", "")
        if chosen:
            self.weights.setText(chosen)
            if not self.new_id.text():
                #: A suggestion, not a decision — an operator names their model
                #: (ADR-0050), and a stem is the best guess available.
                self.new_id.setText(Path(chosen).stem)

    def _register(self) -> None:
        stored = self._session.register_model(
            self.weights.text(),
            model_id=self.new_id.text().strip(),
            task=self.task.currentData(_VALUE),
            framework=self.framework.currentData(_VALUE),
            provenance=self.provenance.text().strip(),
        )
        if stored is not None:
            self.weights.clear()
            self.new_id.clear()
            self.provenance.clear()
            self.reload()

    # ── What is here ──────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Rebuild the table from the project, and re-read which one is active."""
        models = self._session.models()
        active = self._session.active_model
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            here = self._session.model_weights_exist(model)
            for column, text in enumerate(_cells(model, active=active, here=here)):
                item = QTableWidgetItem(text)
                if not here:
                    item.setToolTip(
                        f"No file at {model.path}. The row is real and the weights are "
                        "not — registered on another machine, or moved since."
                    )
                self.table.setItem(row, column, item)
            self.table.item(row, 0).setData(_VALUE, model.model_id)
        self._update_note(models, active)
        self._update_buttons()

    def _update_note(self, models: list[ModelDescriptor], active: str | None) -> None:
        if not models:
            self.note.setText(
                "No models yet. Train one, or register weights you already have below."
            )
            return
        if active is None:
            self.note.setText(
                "No model is in use, so a detector that needs one will refuse before "
                "it reads a scan. Choose one above."
            )
            return
        self.note.setText(f"Detection in this project uses {active}.")

    def _update_buttons(self) -> None:
        selected = self.selected_model()
        self.activate.setEnabled(selected is not None)
        self.detect_with_nothing.setEnabled(self._session.active_model is not None)
        self.register.setEnabled(
            bool(self.weights.text().strip())
            and bool(self.new_id.text().strip())
            and self._session.project is not None
        )

    def selected_model(self) -> str | None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        if len(rows) != 1:
            return None
        item = self.table.item(rows.pop(), 0)
        return None if item is None else str(item.data(_VALUE))

    def _activate_selected(self) -> None:
        self._activate(self.selected_model())

    def _activate(self, model_id: str | None) -> None:
        if self._session.activate_model(model_id):
            self.reload()


def _cells(model: ModelDescriptor, *, active: str | None, here: bool) -> tuple[str, ...]:
    """One model as a row — and the row **is** the comparison (decision 6).

    Everything a reader asks before choosing between two models is a column, and
    none of it is a score: what a model *does* to a scan is M8-T08's report, and
    answering it here from a record would be answering it wrong.
    """
    return (
        model.model_id,
        ACTIVE if model.model_id == active else "",
        str(model.task),
        str(model.framework),
        "" if model.input_size_px is None else str(model.input_size_px),
        ", ".join(model.class_map.values()) or "unnamed",
        model.registered_utc,
        "here" if here else MISSING,
        model.provenance,
    )
