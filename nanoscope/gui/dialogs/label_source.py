"""The one thing a label file cannot say about itself (M7-T09).

A label file is a class index and four numbers. It does not record whether a
person drew the box or a model produced it — and that is precisely the
distinction ADR-0044 made load-bearing: *a model trained on its own output is
confirming itself*.

Which format those numbers are in is **not this layer's business**: the name of
the trainer that reads them belongs to `application` (PROJECT_RULES §2.5, which
is what caught the first draft of this file).

So the import asks, the way `ImportOptions` asks modality and pixel size and
*invents neither* (M5-T07). The operator knows where the file came from; the
application does not, and a default here would be a guess written into every row.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nanoscope.core.entities.project import AnnotationSource

#: What each source means in the sentence an operator is answering, rather than
#: in the vocabulary the database stores it as.
DESCRIPTIONS = {
    AnnotationSource.MANUAL: "drawn by a person",
    AnnotationSource.FROM_DETECTION: "produced by a model",
}


class LabelSource(QDialog):
    """Ask where these labels came from, and hand back the answer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Annotations")

        self.source = QComboBox(self)
        for value, description in DESCRIPTIONS.items():
            self.source.addItem(description, value)

        note = QLabel(
            "A label file does not say who drew the box. Training on boxes a "
            "model produced is a model confirming itself, so the project records "
            "which of the two this was.",
            self,
        )
        note.setWordWrap(True)

        form = QFormLayout()
        form.addRow("These labels were:", self.source)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def choice(self) -> AnnotationSource:
        """What the operator said. There is no default: the dialog is the answer."""
        chosen: AnnotationSource = self.source.currentData()
        return chosen
