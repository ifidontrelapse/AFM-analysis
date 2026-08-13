"""The stored measurement table, beside the particles it describes (M6-T05).

M6's second exit criterion: *"selecting a table row highlights the particle, and
vice versa"*. A run that says a particle is 14 nm tall is a number; the same
number next to the particle it came from is evidence.

**The columns are the producer's own** (ADR-0031). A panel that renamed them for
display would be a second vocabulary, and whoever opened the exported CSV would
find different words for the same measurement.

**A row is linked to a particle by its coordinates, not its index.** The table is
a *subset* of the detections — a height that is not a number is discarded
(ADR-0033) — so row *n* is not detection *n*.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.capabilities import find
from nanoscope.core.entities import AnalysisRun
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: How many digits a float gets on screen. The stored value is untouched; this
#: is the width of a column an operator reads, not a rounding of the data.
DIGITS = 4


class MeasurementsPanel(QWidget):
    """One row per measurement, and a selection that goes both ways."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        #: Detection index per row, by coordinate. Built once per table so a
        #: click does not search the run again.
        self._particles: list[int | None] = []

        #: Which stored run is on screen. Three analyses of one scan leave three
        #: rows, and reaching only the newest is "results persist" satisfied on
        #: a technicality (M6-T09).
        self.run = QComboBox(self)
        self.run.currentIndexChanged.connect(self._run_chosen)

        self.note = QLabel("", self)
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        self.table = QTableWidget(self)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._row_chosen)

        chooser = QHBoxLayout()
        chooser.addWidget(QLabel("Run:", self))
        chooser.addWidget(self.run)
        chooser.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(chooser)
        layout.addWidget(self.note)
        layout.addWidget(self.table)

        session.run_changed.connect(self._run_changed)
        session.particle_selected.connect(self._particle_selected)
        self._run_changed(session.run)

    # ── What it shows ─────────────────────────────────────────────────────────

    def _run_chosen(self, index: int) -> None:
        run_id = self.run.itemData(index)
        if run_id is not None:
            self._session.select_run(int(run_id))

    def _fill_runs(self, run: AnalysisRun | None) -> None:
        """Every stored run of this image, newest last, with the current one shown."""
        self.run.blockSignals(True)
        self.run.clear()
        for stored in self._session.runs():
            self.run.addItem(f"{stored.id}: {stored.mode}, {stored.detector}", stored.id)
        if run is not None:
            self.run.setCurrentIndex(self.run.findData(run.id))
        self.run.blockSignals(False)

    def _run_changed(self, run: AnalysisRun | None) -> None:
        self._fill_runs(run)
        table = self._session.measurements()
        self.table.clearContents()
        self._particles = []

        if run is None:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.note.setText("")
            return
        if table is None:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            #: Not an empty grid: `detect` writes no table at all, and columns
            #: with no rows under them would claim it measured nothing *found*
            #: rather than nothing *asked for* (ADR-0042).
            self.note.setText(f"{run.mode} measured nothing; run a mode that does.")
            return

        note = f"Run {run.id}: {len(table)} measurement(s), {run.detector}"
        #: **The matrix says which modes make masks**, not a literal in a
        #: widget — the panel asking "was this the segmenting one?" by name is
        #: what ADR-0062 removed, and M6-T02's guard caught the attempt.
        capability = find(run.modality.value, run.detector, run.mode)
        if capability is not None and capability.requires_predictor and not run.masks:
            #: A restored segmentation run has detections and no masks, and an
            #: empty overlay reads as *"segmentation found nothing"*. It is not
            #: nothing — it is not stored (ADR-0042, ADR-0064, ADR-0069).
            note += ". Its masks were not stored and cannot be redrawn."
        self.note.setText(note)
        self.table.setColumnCount(len(table.columns))
        self.table.setHorizontalHeaderLabels([str(name) for name in table.columns])
        self.table.setRowCount(len(table))
        for row, (_, record) in enumerate(table.iterrows()):
            for column, name in enumerate(table.columns):
                self.table.setItem(row, column, QTableWidgetItem(_text(record[name])))
            self._particles.append(
                self._session.particle_at(float(record["x_px"]), float(record["y_px"]))
                if {"x_px", "y_px"} <= set(table.columns)
                else None
            )

    # ── Both directions ───────────────────────────────────────────────────────

    def _row_chosen(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        if not rows:
            return
        row = next(iter(rows))
        if 0 <= row < len(self._particles):
            self._session.select_particle(self._particles[row])

    def _particle_selected(self, index: int | None) -> None:
        """Follow a selection made somewhere else — the canvas, so far."""
        if index is None:
            self.table.clearSelection()
            return
        for row, particle in enumerate(self._particles):
            if particle == index:
                #: `blockSignals`, or selecting the row asks the session for the
                #: selection it just announced, which is a loop with two panels
                #: in it.
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
                return


def _text(value: object) -> str:
    """A cell, at a width an operator reads. The stored value is untouched."""
    if isinstance(value, float):
        return f"{value:.{DIGITS}g}"
    return str(value)
