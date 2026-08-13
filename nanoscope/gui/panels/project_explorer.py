"""What is in the project, and what removing it would cost (M5-T04, ADR-0055).

The dock M5-T02 left a placeholder for, and the task ADR-0044 addressed by name:

> *`annotations_for` exists to be counted **before** the deletion, by a
> confirmation dialog that can say "this image has 12 annotations".*

Nothing could remove an image until now, so the obligation had nowhere to land.
This panel can, so this panel pays it — and the dialog says the **count**, not
"are you sure?", because three of the facts involved are non-obvious: annotations
are hand work that cannot be recomputed, the scan file itself is **not** deleted,
and what stays behind becomes an untracked file the integrity check will report.

An image with no annotations is removed without asking. A confirmation that
always appears is one nobody reads by the third time — and then the one that
mattered is clicked through as well.

M5-T06 took the container away from this panel: what it holds now is the session
viewmodel, which is where "which image is selected" lives once more than one
panel cares (ADR-0057). The *asking* stayed here, because how to phrase a
question is a view's decision and a viewmodel that opens a dialog cannot be
tested without a window.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nanoscope.core.entities.project import ImageRecord, OpenedProject
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: Where an `ImageRecord`'s id is kept on its row.
_IMAGE_ID = Qt.ItemDataRole.UserRole


class ProjectExplorer(QWidget):
    """The project's images, and the two things one can do to one of them.

    Subscribes to the viewmodel and to nothing else. It emits no signal of its
    own: a selection is an *intent*, and the panels that care about it hear
    about the result from the session rather than from this widget (ADR-0057).
    """

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        session.project_changed.connect(self.show_project)

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Image", "Scale"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._selection_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)

        self.show_project(session.project)

    # ── What it shows ─────────────────────────────────────────────────────────

    def show_project(self, opened: OpenedProject | None) -> None:
        """Fill the panel, or empty it when a project closes."""
        self.tree.clear()
        if opened is None:
            return

        missing = {image.id for image in opened.integrity.missing_files}
        for image in opened.images:
            self.tree.addTopLevelItem(_row(image, is_missing=image.id in missing))

    @property
    def selected_image_id(self) -> int | None:
        items = self.tree.selectedItems()
        return None if not items else int(items[0].data(0, _IMAGE_ID))

    def _selection_changed(self) -> None:
        image_id = self.selected_image_id
        if image_id is not None:
            self._session.select_image(image_id)

    # ── The one destructive thing it can do ───────────────────────────────────

    def remove_selected(self) -> bool:
        """Remove the selected image, asking first if that would cost something.

        Returns:
            Whether an image was removed — `False` for "nothing selected" and
            for "the operator said no", which the caller does not need to tell
            apart.
        """
        image_id = self.selected_image_id
        image = None if image_id is None else self._session.image_record(image_id)
        if image_id is None or image is None:
            return False

        annotations = self._session.annotation_count(image_id)
        if annotations and not self._confirm(image, annotations):
            return False

        #: The list rebuilds itself from the viewmodel's `project_changed`: the
        #: integrity report is part of what is shown, and the file left behind
        #: has just become untracked (ADR-0040).
        return self._session.remove_image(image_id)

    def _confirm(self, image: ImageRecord, annotations: int) -> bool:
        """Ask, having counted — ADR-0044's obligation, discharged.

        Three facts, because all three are non-obvious: the annotations go, the
        **file does not**, and what stays behind is what the integrity check
        will then call untracked.
        """
        answer = QMessageBox.question(
            self,
            "Remove this image?",
            f"{image.display_name} has {annotations} annotation(s).\n\n"
            "Removing the image deletes them, and they cannot be recomputed.\n"
            f"The file itself stays in {image.relative_path} and will be reported "
            "as untracked.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes


def _row(image: ImageRecord, *, is_missing: bool) -> QTreeWidgetItem:
    """One image as a row: what it is called, and what is known about it.

    A panel that lists an image whose file is gone without saying so is a panel
    that lies quietly, so a missing file is marked in the name column and greyed
    — the report is already in hand (ADR-0040), and not showing it would waste it.
    """
    scale = f"{image.pixel_size_nm:g} nm/px" if image.pixel_size_nm else "scale unknown"
    item = QTreeWidgetItem([image.display_name, scale])
    item.setData(0, _IMAGE_ID, image.id)
    item.setToolTip(0, f"{image.relative_path} ({image.modality})")

    if is_missing:
        item.setText(0, f"{image.display_name} — file missing")
        item.setForeground(0, tokens.qcolor(tokens.WARNING))
        item.setToolTip(0, f"{image.relative_path} is not there; its row is kept (ADR-0040)")
    return item
