"""The two things an import cannot guess (M5-T07, ADR-0083).

**Modality**, because it decides which reader and which measurements apply and
nothing in a filename says it; and **the pixel scale**, because an `.npy` carries
none and a project that records the wrong one produces heights in the wrong
units for the rest of its life.

**A file that states its own scale is not asked about.** Since ADR-0083 the
import reads a Nanoscope header and records what it says, so this field is what
happens to the files that state nothing — and the dialog says so on screen
rather than in a tooltip nobody opens. The operator who reported it put it
plainly: *why is it asking, it should be pulled out while parsing.*

The scale field's minimum reads **"unknown"** rather than `0.00`. Absent is a
value an operator can choose here — the alternative is a blank they have to
trust, and `0` is exactly the fabricated scale ADR-0025 spent a milestone
removing. This is the first surface in the application that *creates* rows, so it
is the one where the invention would start.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nanoscope.core.values import Modality
from nanoscope.gui.theme import tokens

#: What the spin box shows at its minimum. A Nanoscope file carries its own
#: scale in the header and the import reads it (ADR-0083), so "unknown" is not
#: merely allowed for a folder of them — it is the answer that changes nothing.
UNKNOWN = "unknown"

#: How wide the dialog opens, in pixels.
WIDTH_PX = 520


@dataclass(frozen=True)
class ImportChoice:
    """What the operator answered."""

    modality: Modality
    pixel_size_nm: float | None


class ImportOptions(QDialog):
    """Ask the two questions, and hand back the answers."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Images")
        #: Wide enough for the note below to stand on one line: a sentence
        #: explaining why a field is being asked about, wrapped into three, is a
        #: sentence that reads like fine print.
        self.setMinimumWidth(WIDTH_PX)

        self.modality = QComboBox(self)
        for value in Modality:
            self.modality.addItem(value.value.upper(), value)

        self.pixel_size = QDoubleSpinBox(self)
        self.pixel_size.setRange(0.0, 10_000.0)
        self.pixel_size.setDecimals(4)
        self.pixel_size.setSuffix(" nm/px")
        self.pixel_size.setSpecialValueText(UNKNOWN)
        self.pixel_size.setToolTip(
            "Used only for the files that state no scale of their own — an .npy, "
            "or an SEM/TEM image.\n"
            "A Nanoscope file (.spm, .000, .001, …) carries its scan size in its "
            "header, and that is what the project records (ADR-0083).\n"
            "'unknown' is a legitimate answer: physical sizes are then absent "
            "rather than wrong."
        )

        #: On screen, not in a tooltip: the question *"why is it asking me
        #: this?"* is the one the field caused, and a tooltip answers only the
        #: operator who already suspected there was something to hover over.
        self.note = QLabel(
            "A Nanoscope file states its own scale; the header wins over this field.", self
        )
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        form = QFormLayout()
        form.addRow("Modality:", self.modality)
        form.addRow("Pixel size:", self.pixel_size)
        form.addRow("", self.note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def choice(self) -> ImportChoice:
        """The answers. `pixel_size_nm` is `None` at the "unknown" end."""
        scale = self.pixel_size.value()
        return ImportChoice(
            modality=self.modality.currentData(),
            pixel_size_nm=scale if scale > 0 else None,
        )
