"""What the selected scan is, in numbers (M5-T06).

The dock M5-T02 left a placeholder for, and the second consumer that justified
the viewmodel: it describes the array **the viewer is already showing**, read
from the session rather than from the file. Reading it twice would cost a disk
read per selection and — worse — make it possible for the two panels to disagree
about the same scan.

Every field is either in the file or derived from it. Nothing is inferred: an
image with no scale has no physical size and says so, which is ADR-0025's rule at
the third surface in this milestone.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from nanoscope.application.use_cases.display import DisplayImage, value_range
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: The fields, in the order an operator reads them: what it is, how big it is,
#: and what is in it.
FIELDS = ("Name", "Modality", "Size", "Physical size", "Pixel size", "Data type", "Value range")

#: Shown when nothing is selected, and for a field that cannot be answered.
ABSENT = "—"


class PropertiesPanel(QWidget):
    """One image, described. Holds the viewmodel and no data of its own."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        form = QFormLayout()
        self.values = {name: QLabel(ABSENT, self) for name in FIELDS}
        for name in FIELDS:
            form.addRow(f"{name}:", self.values[name])

        #: An empty section is a promise when it names the task that fills it,
        #: and a bug when it does not (M5-T02's rule, applied inside a panel).
        runs = QLabel("Analysis runs appear here in M6.", self)
        runs.setWordWrap(True)
        runs.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(runs)
        layout.addStretch(1)

        session.image_changed.connect(self.show_image)
        self.show_image(session.image)

    def show_image(self, image: DisplayImage | None) -> None:
        """Describe the loaded image, or say there is none."""
        for name, value in _describe(image).items():
            self.values[name].setText(value)


def _describe(image: DisplayImage | None) -> dict[str, str]:
    """The fields as text — a pure function, so the wording is testable.

    Kept out of the widget because *what* the numbers say is worth a test and
    *where the label sits* is not.
    """
    if image is None:
        return dict.fromkeys(FIELDS, ABSENT)

    width, height = image.size_px
    physical = image.size_nm
    low, high = value_range(image, full=True)
    return {
        "Name": image.name,
        #: `AFM`, not `afm`: the enum's value is a storage token, and three
        #: instrument names that are acronyms everywhere else read as typos.
        "Modality": image.modality.value.upper(),
        "Size": f"{width} x {height} px",
        #: No scale, no size — never a fabricated one (ADR-0025).
        "Physical size": (
            f"{physical[0]:.1f} x {physical[1]:.1f} nm" if physical else "scale unknown"
        ),
        "Pixel size": (
            f"{image.pixel_size_nm:g} nm/px" if image.pixel_size_nm is not None else "unknown"
        ),
        "Data type": str(image.data.dtype),
        #: Units are deliberately absent: this is the raw array, and calling a
        #: raw z value "nm" is the kind of invention this project keeps deleting.
        "Value range": f"{low:.4g} … {high:.4g}",
    }
