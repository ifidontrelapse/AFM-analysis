"""Which files, with a look at them first (2026-09-02).

The operator asked for a preview when picking an image, and the request is one
sentence long because the problem is: an acquisition off a Bruker is called
`2-6-dmfa-pvp.039`, the folder holds forty of them, and the name says the
acquisition number and nothing about what was scanned. Picking the right six out
of forty meant importing all forty and looking afterwards.

So this is Qt's file dialog with a pane beside it — **not** a chooser this
project wrote. A file dialog is places, filters, keyboard navigation, typing a
path, sorting by date; reimplementing it to gain a preview would trade all of
that for one label. What it costs is `DontUseNativeDialog`: the pane can only be
added to Qt's own widget, so the dialog stops being the desktop's. On a window
already carrying one dark theme of its own (ADR-0002) that is a small price, and
it is the same trade Gwyddion and every SPM tool with a preview makes.

**The preview reads the file, and reads it the way the viewer will.** Same
loader, same colormap, same rule about an unknown scale — because the point of
looking before importing is that what is on screen is what lands in the project.
It shows the scale a Nanoscope header states (ADR-0083), which is also the
answer to *"why is it asking me for a pixel size?"* — for these files it is not.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.use_cases.display import DisplayImage, load_file_for_display, thumbnail
from nanoscope.core.errors import NanoscopeError
from nanoscope.gui.pixmaps import to_pixmap
from nanoscope.gui.theme import tokens

#: How big the rendered preview is, before it is scaled into the pane. Bigger
#: than the pane so a scaled-down picture stays sharp, small enough that a
#: 4096² scan does not cost a second of colormapping per arrow-key press.
PREVIEW_PX = 320

#: The pane's width. A file dialog that suddenly needs half the screen is one an
#: operator resizes back every time.
PANE_PX = 260

#: How wide the dialog opens. Qt's default width is picked for a dialog with
#: three columns, and this one has a fourth: at that width the pane takes its
#: PANE_PX out of the file list, which is the column an operator is reading.
WIDTH_PX = 1000

#: Shown before anything is highlighted, and for a directory.
NOTHING_SELECTED = "Select a file to preview it."


class ImageChooser(QFileDialog):
    """Qt's open-files dialog, with a preview pane on the right.

    The pane is a child of Qt's own grid layout, which exists only in the
    non-native dialog — so `DontUseNativeDialog` is not a preference here, it is
    the precondition, and it is set in the constructor rather than left to a
    caller who would forget once.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, "Import Images")
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)
        self.resize(WIDTH_PX, self.height())

        self.picture = QLabel(self)
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setMinimumSize(PANE_PX, PANE_PX)

        self.facts = QLabel(NOTHING_SELECTED, self)
        self.facts.setWordWrap(True)
        self.facts.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        pane = QWidget(self)
        pane.setMaximumWidth(PANE_PX)
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(tokens.SPACE_MD, 0, 0, 0)
        layout.addWidget(self.picture)
        layout.addWidget(self.facts)
        layout.addStretch(1)

        #: Qt lays the non-native dialog out in a 3-column grid; the pane goes in
        #: a fourth, spanning the rows the file list occupies. `layout()` is
        #: typed as the abstract base — this one is the grid, and if a Qt release
        #: ever changes that, the dialog opens without a pane instead of raising
        #: at a modal call site.
        grid = self.layout()
        if isinstance(grid, QGridLayout):
            grid.addWidget(pane, 1, grid.columnCount(), grid.rowCount() - 1, 1)

        self.currentChanged.connect(self.show_preview)

    def show_preview(self, path: str) -> None:
        """Draw the highlighted file, or say why it cannot be drawn.

        Every refusal this can meet is a `NanoscopeError` with a sentence in it
        (ADR-0030) — *"no AFM reader for notes.txt"*, *"Ciao image list blocks
        not found"* — and a preview pane is exactly the place to show it: the
        alternative is importing the file to find out.

        Args:
            path: what the dialog is highlighting; empty for a directory or for
                nothing at all.
        """
        file = Path(path)
        if not path or not file.is_file():
            self.picture.setPixmap(QPixmap())
            self.facts.setText(NOTHING_SELECTED)
            return

        try:
            image = load_file_for_display(file)
            picture = to_pixmap(thumbnail(image, size_px=PREVIEW_PX))
        except NanoscopeError as refusal:
            self.picture.setPixmap(QPixmap())
            self.facts.setText(f"{file.name}\nno preview: {refusal}")
            return

        self.picture.setPixmap(
            picture.scaled(
                self.picture.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.facts.setText(describe(image))

    def chosen(self) -> list[str]:
        """The files the operator selected — empty when they cancelled."""
        return self.selectedFiles()


def describe(image: DisplayImage) -> str:
    """The three facts a name does not carry, as text — a pure function.

    The scale is the one that answers a question the operator asked out loud:
    **where it comes from** is part of the fact, because *"1.95 nm/px"* beside a
    dialog that is about to ask for a pixel size reads as a contradiction, and
    *"from the file's header"* reads as the answer (ADR-0083).
    """
    width, height = image.size_px
    physical = image.size_nm
    lines = [image.name, f"{width} x {height} px"]
    if image.pixel_size_nm is None:
        #: ADR-0025 at the surface that comes *before* a row exists: a preview
        #: that invented a scale here would be inventing it for the import.
        lines.append("no scale in the file — state one in the next dialog")
    else:
        lines.append(f"{image.pixel_size_nm:g} nm/px, from the file's header")
    if physical is not None:
        lines.append(f"{physical[0]:.0f} x {physical[1]:.0f} nm")
    return "\n".join(lines)
