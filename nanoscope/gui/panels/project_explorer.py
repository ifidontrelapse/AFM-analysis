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

**Since 2026-09-02 the rows carry a thumbnail**, for the same reason the import
dialog got a preview: a Bruker names its files by acquisition number, so a list
of names is a list of numbers. They are drawn **one per turn of the event loop**
and never all at once — forty scans is forty file reads, and a list that reads
forty files before it appears is a window that hangs on open. The queue is
abandoned when the project changes, so opening a second project does not finish
drawing the first.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.settings import COLORMAP_SETTING
from nanoscope.application.use_cases.display import COLORMAPS, thumbnail
from nanoscope.core.entities.project import ImageRecord, OpenedProject
from nanoscope.gui.pixmaps import to_pixmap
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: Where an `ImageRecord`'s id is kept on its row.
_IMAGE_ID = Qt.ItemDataRole.UserRole

#: How big a row's picture is. Two rows of text tall — enough to tell a dense
#: field from an empty scan, which is all a list is being asked to do.
THUMBNAIL_PX = 32


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
        #: A panel listing the images while a *different* one is on screen is a
        #: panel that lies, so the row follows a selection made elsewhere —
        #: M6-T08's Next/Previous, so far.
        session.image_changed.connect(lambda _image: self.follow_selection())

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Image", "Scale"])
        self.tree.setRootIsDecorated(False)
        self.tree.setIconSize(QSize(THUMBNAIL_PX, THUMBNAIL_PX))
        #: The name takes what is left, the scale takes what it needs. Left to
        #: Qt's defaults the two columns split the dock evenly, and a 32-pixel
        #: icon then pushes `2-6-dmfa-pvp.039` into `2-6-d…` — the picture and
        #: the name are both what the row is for.
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._selection_changed)

        #: The rows still waiting for a picture, oldest first. A list rather
        #: than a set: they are drawn in the order they are shown, so what the
        #: operator is looking at fills in first.
        self._pending: list[int] = []

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

        #: A file that is not there has no picture, and asking for one is a read
        #: that fails per row (ADR-0040 already reported it).
        self._pending = [image.id for image in opened.images if image.id not in missing]
        QTimer.singleShot(0, self._draw_one_thumbnail)

    def follow_selection(self) -> None:
        """Show the session's selection, without announcing it back.

        `blockSignals`, or setting the row asks the session for the selection it
        has just announced — the loop M6-T05 met once already on the
        measurements table.
        """
        image_id = self._session.image_id
        if image_id is None or image_id == self.selected_image_id:
            return
        item = self._item_for(image_id)
        if item is not None:
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(item)
            self.tree.blockSignals(False)

    @property
    def selected_image_id(self) -> int | None:
        items = self.tree.selectedItems()
        return None if not items else int(items[0].data(0, _IMAGE_ID))

    def _selection_changed(self) -> None:
        image_id = self.selected_image_id
        if image_id is not None:
            self._session.select_image(image_id)

    # ── The pictures, one turn of the event loop at a time ────────────────────

    def draw_next_thumbnail(self) -> bool:
        """Draw one queued row's picture.

        One, not all: reading a file takes milliseconds and forty of them takes
        long enough to see, and a panel that blocks while it draws itself is the
        one an operator meets before anything else in the window.

        Returns:
            Whether any rows are still waiting — what the pump reschedules on,
            and what a test loops on instead of running an event loop.
        """
        while self._pending:
            image_id = self._pending.pop(0)
            item = self._item_for(image_id)
            if item is None:
                #: The project changed under us between the queue and the row.
                continue
            image = self._session.read_image(image_id)
            if image is None:
                #: Unreadable is not empty: the row keeps its name and its
                #: tooltip, and the viewer is where the refusal is worth a
                #: sentence (ADR-0030).
                continue
            item.setIcon(
                0,
                QIcon(to_pixmap(thumbnail(image, size_px=THUMBNAIL_PX, colormap=self._colormap()))),
            )
            break
        return bool(self._pending)

    def _draw_one_thumbnail(self) -> None:
        """The pump: draw one, and come back for the next one *later*."""
        if self.draw_next_thumbnail():
            QTimer.singleShot(0, self._draw_one_thumbnail)

    def _colormap(self) -> str:
        """The operator's default colormap, or the first one.

        The same preference the viewer opens a scan with (M5-T09), so a row and
        the canvas above it are the same picture. A stored value this version
        does not offer is ignored rather than raised on — `render` would refuse
        it, and a settings file is not worth an empty list.
        """
        stored = str(self._session.preference(COLORMAP_SETTING, COLORMAPS[0]))
        return stored if stored in COLORMAPS else COLORMAPS[0]

    def _item_for(self, image_id: int) -> QTreeWidgetItem | None:
        """The row carrying that id, or `None` when the list has moved on."""
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item is not None and int(item.data(0, _IMAGE_ID)) == image_id:
                return item
        return None

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
