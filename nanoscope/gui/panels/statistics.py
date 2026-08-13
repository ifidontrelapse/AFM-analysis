"""The run as a distribution, not as thirty rows (M6-T06, ADR-0066).

The numbers come from `application.use_cases.statistics`; what is here is a
column choice, six labels and a bar chart painted by Qt.

**Painted by Qt on purpose.** matplotlib lives in `infrastructure` and `gui/` may
not import it (Architecture §3.2, checked by a test since M5-T06). The binning is
`numpy`'s, in `application`; what is left for a widget is rectangles, and
`QtCharts` would be a whole module to draw a chart nobody interacts with.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.use_cases.statistics import (
    Summary,
    histogram,
    numeric_columns,
    summarise,
)
from nanoscope.core.entities import AnalysisRun
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: What the summary reports, in the order it is read: how many, where the middle
#: is, how wide the spread, and where the ends are.
FIELDS = ("Particles", "Mean", "Median", "Std dev", "Minimum", "Maximum")


class Histogram(QWidget):
    """Counts as bars. No axes, no legend — the numbers are above it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._counts: np.ndarray = np.zeros(0, dtype=int)
        self.setMinimumHeight(90)

    def show_bins(self, counts: np.ndarray, _edges: np.ndarray) -> None:
        self._counts = counts
        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 — Qt's name
        if self._counts.size == 0 or self._counts.max() == 0:
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), tokens.qcolor(tokens.BACKGROUND))
        painter.setBrush(tokens.qcolor(tokens.ACCENT))
        painter.setPen(Qt.PenStyle.NoPen)

        width = self.width() / self._counts.size
        tallest = float(self._counts.max())
        for index, count in enumerate(self._counts):
            height = self.height() * float(count) / tallest
            painter.drawRect(
                QRectF(index * width, self.height() - height, max(width - 1.0, 1.0), height)
            )
        painter.end()


class StatisticsPanel(QWidget):
    """One column of one run, described and drawn."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session

        self.column = QComboBox(self)
        self.column.currentTextChanged.connect(lambda _text: self._describe())

        self.values = {name: QLabel("—", self) for name in FIELDS}
        form = QFormLayout()
        for name in FIELDS:
            form.addRow(f"{name}:", self.values[name])

        self.note = QLabel("", self)
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        self.chart = Histogram(self)

        chooser = QHBoxLayout()
        chooser.addWidget(QLabel("Column:", self))
        chooser.addWidget(self.column)
        chooser.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(chooser)
        layout.addLayout(form)
        layout.addWidget(self.chart)
        layout.addWidget(self.note)

        session.run_changed.connect(self._run_changed)
        self._run_changed(session.run)

    def _run_changed(self, run: AnalysisRun | None) -> None:
        table = self._session.measurements()
        self.column.blockSignals(True)
        self.column.clear()
        if table is not None:
            self.column.addItems(numeric_columns(table))
        self.column.blockSignals(False)

        if run is None:
            self.note.setText("")
        elif table is None:
            self.note.setText(f"{run.mode} measured nothing to describe.")
        elif run.pixel_size_nm is None:
            #: **Lateral**, and the word matters: a height is calibrated by the
            #: z axis and stays in nanometres, while a radius comes from the
            #: pixel size and is absent without it. Saying "no physical columns"
            #: here would be wrong about half the table (ADR-0025, and the
            #: reason this panel found out).
            self.note.setText(
                "The lateral scale is unknown, so sizes in nanometres are absent. "
                "Heights are calibrated by the z axis and are not affected."
            )
        else:
            self.note.setText("")
        self._describe()

    def _describe(self) -> None:
        table = self._session.measurements()
        column = self.column.currentText()
        summary = None if table is None or not column else summarise(table, column)

        for name in FIELDS:
            self.values[name].setText(_field(summary, name))
        if table is not None and column:
            self.chart.show_bins(*histogram(table, column))
        else:
            self.chart.show_bins(np.zeros(0, dtype=int), np.zeros(0))


def _field(summary: Summary | None, name: str) -> str:
    """One number, at a width an operator reads. Absent stays absent."""
    if summary is None:
        return "—"
    value = {
        "Particles": summary.count,
        "Mean": summary.mean,
        "Median": summary.median,
        "Std dev": summary.std,
        "Minimum": summary.minimum,
        "Maximum": summary.maximum,
    }[name]
    if isinstance(value, int):
        return str(value)
    #: `nan` is what a standard deviation of one particle *is*; printing it as a
    #: number would be a spread nobody measured.
    return "—" if not np.isfinite(value) else f"{value:.4g}"
