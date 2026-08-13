"""Drawing a box, and the label that makes it an annotation (M7-T02, ADR-0071).

The first surface in this project where an operator **makes** data rather than
asking for it to be computed. Everything it produces goes through the command
stack, because M4-T08 built undo for exactly this and a drawing tool without undo
is one nobody can afford to be quick with.

**There is no point tool**, and the reason is the shape: ADR-0044 stores a box
and refuses a zero-area one twice. A point has no extent, so a point tool must
invent one — and a "point size" control is that invention wearing a label.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nanoscope.core.entities.project import Annotation, Ruler, RulerKind
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: What a new box is called when the operator has not said. Deliberately not
#: empty and deliberately generic: it is a label they will want to change, which
#: is a different situation from one they never gave (ADR-0070).
DEFAULT_LABEL = "particle"


class AnnotatePanel(QWidget):
    """A label, a tool that draws boxes with it, and what has been drawn."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session

        self.label = QLineEdit(DEFAULT_LABEL, self)
        self.label.setToolTip(
            "Applied to every box drawn from now on. Annotating forty particles "
            "through forty dialogs is a feature nobody uses twice."
        )

        self.draw = QPushButton("Draw boxes", self)
        self.draw.setCheckable(True)
        self.draw.setToolTip("While this is on, dragging draws a box instead of panning the scan.")
        self.draw.toggled.connect(self._toggled)

        self.outline = QPushButton("Draw outlines", self)
        self.outline.setCheckable(True)
        self.outline.setToolTip(
            "Click to add a vertex, double-click to close. A particle that is "
            "not a rectangle is the ordinary case in this science."
        )
        self.outline.toggled.connect(self._outline_toggled)

        self.brush = QPushButton("Paint masks", self)
        self.brush.setCheckable(True)
        self.brush.setToolTip(
            "Drag to paint. The mask is stored as a file beside the project's "
            "database (PROJECT_RULES §5); the scan itself is never touched."
        )
        self.brush.toggled.connect(self._brush_toggled)

        self.brush_size = QSpinBox(self)
        self.brush_size.setRange(1, 200)
        self.brush_size.setValue(8)
        self.brush_size.setSuffix(" px")

        self.measure = QPushButton("Measure distance", self)
        self.measure.setCheckable(True)
        self.measure.setToolTip(
            "Drag between two points. The length is stored as the two points, "
            "never as a number — a stored length is a second answer waiting to "
            "disagree (ADR-0074)."
        )
        self.measure.toggled.connect(self._measure_toggled)

        self.as_profile = QCheckBox("…as a profile line", self)
        self.as_profile.setToolTip(
            "The same line, read as the heights underneath it rather than as a "
            "length. One geometry, two readings (ADR-0074)."
        )

        self.lengths = QLabel("", self)
        self.lengths.setWordWrap(True)
        self.lengths.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        self.report = QLabel("", self)
        self.report.setWordWrap(True)
        self.report.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Label:", self))
        layout.addWidget(self.label)
        layout.addWidget(self.draw)
        layout.addWidget(self.outline)
        layout.addWidget(self.brush)
        layout.addWidget(self.brush_size)
        layout.addWidget(self.measure)
        layout.addWidget(self.as_profile)
        layout.addWidget(self.lengths)
        layout.addWidget(self.report)
        layout.addStretch(1)

        session.annotations_changed.connect(self._annotations_changed)
        session.rulers_changed.connect(self._rulers_changed)
        session.image_changed.connect(lambda _image: self._update())
        self._annotations_changed(session.annotations)
        self._rulers_changed(session.rulers)

    def box_drawn(self, box: tuple[float, float, float, float]) -> None:
        """Store what was dragged, with the label the field is showing."""
        self._session.add_annotation(box, label=self.label.text())

    def polygon_drawn(self, points: tuple[tuple[float, float], ...]) -> None:
        """Store the outline that was just closed, with the field's label."""
        self._session.add_polygon(points, label=self.label.text())

    def mask_painted(self, mask: object) -> None:
        """Store the stroke that was just lifted, with the field's label."""
        self._session.add_mask(mask, label=self.label.text())  # type: ignore[arg-type]

    def line_drawn(self, line: tuple[tuple[float, float], tuple[float, float]]) -> None:
        """Store the line that was just dragged, with the field's label."""
        start, end = line
        self._session.add_ruler(
            start,
            end,
            label=self.label.text(),
            kind=RulerKind.PROFILE if self.as_profile.isChecked() else RulerKind.DISTANCE,
        )

    def _rulers_changed(self, rulers: tuple[Ruler, ...]) -> None:
        """What has been measured, in pixels and — when there is a scale — in
        nanometres. Without one the words say so rather than a number implying
        a scale nobody recorded (ADR-0025)."""
        lines = []
        for ruler in rulers:
            px, nm = self._session.ruler_length(ruler)
            length = f"{px:.1f} px" if nm is None else f"{nm:.1f} nm ({px:.1f} px)"
            lines.append(f"{ruler.label}: {length}")
        if rulers and all(self._session.ruler_length(one)[1] is None for one in rulers):
            lines.append("scale unknown, so lengths are in pixels only")
        self.lengths.setText("\n".join(lines))

    def _measure_toggled(self, on: bool) -> None:
        self.measure.setText("Measuring" if on else "Measure distance")
        if on:
            for other in (self.draw, self.outline, self.brush):
                other.setChecked(False)

    def _toggled(self, on: bool) -> None:
        self.draw.setText("Drawing boxes" if on else "Draw boxes")
        #: One tool at a time: two drawing modes on one canvas is a gesture that
        #: means two things.
        if on:
            self.outline.setChecked(False)
            self.brush.setChecked(False)
            self.measure.setChecked(False)

    def _outline_toggled(self, on: bool) -> None:
        self.outline.setText("Drawing outlines" if on else "Draw outlines")
        if on:
            self.draw.setChecked(False)
            self.brush.setChecked(False)
            self.measure.setChecked(False)

    def _brush_toggled(self, on: bool) -> None:
        self.brush.setText("Painting masks" if on else "Paint masks")
        if on:
            self.draw.setChecked(False)
            self.outline.setChecked(False)
            self.measure.setChecked(False)

    def _annotations_changed(self, annotations: tuple[Annotation, ...]) -> None:
        self.report.setText(
            "Nothing drawn on this scan yet."
            if not annotations
            else f"{len(annotations)} annotation(s) on this scan."
        )
        self._update()

    def _update(self) -> None:
        can_draw = self._session.image_id is not None
        for tool in (self.draw, self.outline, self.brush, self.measure):
            tool.setEnabled(can_draw)
            if not can_draw:
                tool.setChecked(False)
