"""The two things an import cannot guess (M5-T07).

**Modality**, because it decides which reader and which measurements apply and
nothing in a filename says it; and **the pixel scale**, because an `.npy` carries
none and a project that records the wrong one produces heights in the wrong
units for the rest of its life.

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
    QVBoxLayout,
    QWidget,
)

from nanoscope.core.values import Modality

#: What the spin box shows at its minimum. `.spm` files carry their own scale in
#: the header, so "unknown" is the right answer for a folder of them too.
UNKNOWN = "unknown"


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

        self.modality = QComboBox(self)
        for value in Modality:
            self.modality.addItem(value.value.upper(), value)

        self.pixel_size = QDoubleSpinBox(self)
        self.pixel_size.setRange(0.0, 10_000.0)
        self.pixel_size.setDecimals(4)
        self.pixel_size.setSuffix(" nm/px")
        self.pixel_size.setSpecialValueText(UNKNOWN)
        self.pixel_size.setToolTip(
            "Leave it at 'unknown' when the file carries its own scale (.spm) or "
            "when nobody recorded one. Physical sizes are then absent rather than wrong."
        )

        form = QFormLayout()
        form.addRow("Modality:", self.modality)
        form.addRow("Pixel size:", self.pixel_size)

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
