"""The heights under a drawn line (M7-T06, ADR-0075).

M7's third exit criterion names the notebook as its reference, and the notebook
plots one thing: `z_flat[y, x1:x2]` against distance in nanometres. This is that
plot, for a line an operator drew anywhere.

**It names the stage it measured.** Profiling a raw map and a flattened one give
different numbers, and a measurement whose provenance is a checkbox somebody set
four clicks ago is not a measurement anybody can defend.

Painted by Qt for the same reason as M6-T06's histogram: matplotlib lives in
`infrastructure`, and `gui/` may not import it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from nanoscope.application.use_cases.display import STAGE_LABELS
from nanoscope.core.entities import Ruler
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel


class ProfileView(QWidget):
    """One polyline, scaled to the widget. No axes — the numbers are above it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._heights: np.ndarray = np.zeros(0)
        self.setMinimumHeight(110)

    def show_profile(self, heights: np.ndarray) -> None:
        self._heights = np.asarray(heights, dtype=float)
        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 — Qt's name
        finite = self._heights[np.isfinite(self._heights)] if self._heights.size else self._heights
        if finite.size < 2:
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), tokens.qcolor(tokens.BACKGROUND))
        pen = QPen(tokens.qcolor(tokens.ACCENT))
        pen.setWidthF(1.5)
        painter.setPen(pen)

        low, high = float(finite.min()), float(finite.max())
        span = high - low or 1.0
        step = self.width() / max(self._heights.size - 1, 1)
        points = [
            QPointF(index * step, self.height() * (1.0 - (value - low) / span))
            for index, value in enumerate(self._heights)
            if np.isfinite(value)
        ]
        painter.drawPolyline(QPolygonF(points))
        painter.end()


class ProfilePanel(QWidget):
    """Pick one of the lines drawn on this scan, and read what is under it."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session

        self.ruler = QComboBox(self)
        self.ruler.currentIndexChanged.connect(lambda _index: self._describe())

        self.summary = QLabel("", self)
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        self.view = ProfileView(self)

        chooser = QHBoxLayout()
        chooser.addWidget(QLabel("Line:", self))
        chooser.addWidget(self.ruler)
        chooser.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(chooser)
        layout.addWidget(self.view)
        layout.addWidget(self.summary)

        session.rulers_changed.connect(self._rulers_changed)
        session.image_changed.connect(lambda _image: self._describe())
        session.preview_changed.connect(lambda _preview: self._describe())
        self._rulers_changed(session.rulers)

    def _rulers_changed(self, rulers: tuple[Ruler, ...]) -> None:
        self.ruler.blockSignals(True)
        self.ruler.clear()
        for one in rulers:
            self.ruler.addItem(f"{one.id}: {one.label}", one.id)
        self.ruler.blockSignals(False)
        self.ruler.setCurrentIndex(self.ruler.count() - 1)
        self._describe()

    def _describe(self) -> None:
        chosen = next(
            (one for one in self._session.rulers if one.id == self.ruler.currentData()), None
        )
        if chosen is None:
            self.view.show_profile(np.zeros(0))
            self.summary.setText("Draw a line to read the heights under it.")
            return

        profile = self._session.ruler_profile(chosen)
        if profile is None:
            self.view.show_profile(np.zeros(0))
            self.summary.setText("No scan on screen to measure.")
            return

        distances, nm, heights = profile
        self.view.show_profile(heights)
        length = f"{distances[-1]:.1f} px" if nm is None else f"{nm[-1]:.1f} nm"
        finite = heights[np.isfinite(heights)]
        span = "" if finite.size == 0 else f", {finite.min():.4g} to {finite.max():.4g}"
        #: The stage is part of the answer: a raw map and a flattened one give
        #: different numbers, and both are legitimate questions (ADR-0075).
        self.summary.setText(
            f"{chosen.label}: {len(heights)} samples over {length}{span} — "
            f"measured on {STAGE_LABELS[self._session.stage]}"
        )
