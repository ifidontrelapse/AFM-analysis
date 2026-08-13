"""The scan on screen, with the numbers that make it a measurement (M5-T05).

M5's second exit criterion: *"a scan renders with correct nm axes and a scale
bar"*. Three things separate this from a picture viewer, and all three are here:

- a **scale bar** in nanometres, sized to a round number and redrawn on zoom;
- a **readout** in nm *and* px, with the value under the cursor;
- an honest answer when the scale is unknown — no bar, pixels only, and the
  words "scale unknown". ADR-0025 spent a milestone on absent-not-fabricated,
  and a viewer inventing 1 nm/px would undo it in one line.

`QGraphicsView` rather than matplotlib: zoom and pan are what a graphics view
*is*, and a matplotlib canvas re-renders on every wheel event, which is why
scientific GUIs feel slow. matplotlib stays in `infrastructure/imaging/plots.py`,
for figures that get saved.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPixmap, QResizeEvent, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.use_cases.display import (
    COLORMAPS,
    DisplayImage,
    render,
    value_range,
)
from nanoscope.gui.viewmodels import SessionViewModel

#: How far in and out the wheel may go. Not taste: past 64x a pixel fills the
#: window and there is nothing more to see, and below 1/32 the scan is a dot.
MIN_ZOOM, MAX_ZOOM = 1 / 32, 64.0

#: A scale bar is only useful at a round length. These are the lengths it may
#: take, in nm, and it picks the largest that fits in a fifth of the view.
BAR_LENGTHS_NM = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000)


class ImageView(QGraphicsView):
    """A pannable, zoomable canvas that says where the cursor is."""

    #: `(x_px, y_px)` under the cursor, or `None` when it leaves the image.
    hovered = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self.scene().addItem(self._item)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self._zoom = 1.0
        #: Whether the view is still showing the whole scan. A resize refits
        #: while it is, and leaves the operator's zoom alone once it is not.
        self._fitted = True

    def show_pixmap(self, pixmap: QPixmap) -> None:
        self._item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.reset_zoom()

    def clear(self) -> None:
        self._item.setPixmap(QPixmap())
        self.setSceneRect(QRectF())

    @property
    def zoom(self) -> float:
        return self._zoom

    def reset_zoom(self) -> None:
        """Fit the whole scan in the view — where an operator starts."""
        self.resetTransform()
        self._zoom = 1.0
        self._fitted = True
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt's name
        """Refit while the whole scan is showing.

        Without this the scan is a postage stamp: `fitInView` at load time runs
        before the widget has its final size, so it fits the image to a layout
        that has not happened yet. Only while `_fitted` — once the operator has
        zoomed, a resize must not throw their view away.
        """
        super().resizeEvent(event)
        if self._fitted:
            self.reset_zoom()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 — Qt's name
        """Zoom about the cursor, within limits.

        Clamped rather than unbounded: past `MAX_ZOOM` one pixel fills the
        window and there is nothing further to see, and the transform loses
        precision long before that becomes interesting.
        """
        if self._item.pixmap().isNull():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        target = self._zoom * factor
        if not MIN_ZOOM <= target <= MAX_ZOOM:
            return
        self._zoom = target
        self._fitted = False
        self.scale(factor, factor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt's name
        super().mouseMoveEvent(event)
        point = self.mapToScene(event.position().toPoint())
        if self._item.pixmap().isNull() or not self.sceneRect().contains(point):
            self.hovered.emit(None)
            return
        self.hovered.emit((int(point.x()), int(point.y())))


class ImageViewer(QWidget):
    """The panel: the canvas, the colormap, the value window, and the scale bar."""

    #: What to put in the status bar. The window owns the status bar; this owns
    #: the sentence.
    readout = Signal(str)

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: DisplayImage | None = None
        session.image_changed.connect(self.show_image)

        self.view = ImageView(self)
        self.view.hovered.connect(self._describe)

        self.colormap = QComboBox(self)
        self.colormap.addItems(COLORMAPS)
        self.colormap.currentTextChanged.connect(lambda _: self._redraw())

        self.full_range = QCheckBox("Full range", self)
        self.full_range.setToolTip(
            "Off: the 2nd-98th percentile, so one hot pixel does not flatten the image.\n"
            "On: every value, so nothing is clipped."
        )
        self.full_range.toggled.connect(lambda _: self._redraw())

        self.scale_label = QLabel("", self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Colormap", self))
        controls.addWidget(self.colormap)
        controls.addWidget(self.full_range)
        controls.addStretch(1)
        controls.addWidget(self.scale_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.view)

        self.show_image(session.image)

    # ── What it shows ─────────────────────────────────────────────────────────

    def show_image(self, image: DisplayImage | None) -> None:
        """Draw what the session loaded, or empty the canvas.

        The loading happened in the viewmodel — a widget that reads a file
        decides *what* to read, which is the line Architecture §2.3 draws. What
        is left here is the part that is genuinely presentation: an array, a
        colormap, and a pixmap.
        """
        self._image = image
        if image is None:
            self.view.clear()
            self.scale_label.setText("")
            return

        self._redraw()
        self.readout.emit(self._summary())

    def _redraw(self) -> None:
        if self._image is None:
            return
        limits = value_range(self._image, full=self.full_range.isChecked())
        rgb = render(self._image, colormap=self.colormap.currentText(), limits=limits)
        self.view.show_pixmap(QPixmap.fromImage(_to_qimage(rgb)))
        self.scale_label.setText(self._scale_bar_text())

    # ── The numbers that make it a measurement ────────────────────────────────

    def _summary(self) -> str:
        image = self._image
        if image is None:  # pragma: no cover — only called with one loaded
            return ""
        width, height = image.size_px
        physical = image.size_nm
        extent = f"{physical[0]:.0f} x {physical[1]:.0f} nm" if physical else "scale unknown"
        return f"{image.name}: {width} x {height} px, {extent}"

    def _scale_bar_text(self) -> str:
        """The bar, as a label. Absent when the scale is.

        A viewer that draws a bar without a scale is a viewer inventing one —
        the exact substitution ADR-0025 spent a milestone removing.
        """
        image = self._image
        if image is None or image.pixel_size_nm is None:
            return "scale unknown"
        return f"▬ {_bar_length_nm(image, self.view.zoom):g} nm"

    def _describe(self, position: tuple[int, int] | None) -> None:
        """Where the cursor is, in nm *and* px, with the value under it."""
        image = self._image
        if image is None:
            return
        if position is None:
            self.readout.emit(self._summary())
            return

        x, y = position
        height, width = image.data.shape[:2]
        if not (0 <= x < width and 0 <= y < height):  # pragma: no cover — clamped by the view
            return

        value = float(image.data[y, x])
        where = f"x={x} y={y} px"
        if image.pixel_size_nm is not None:
            where += f"  ({x * image.pixel_size_nm:.1f}, {y * image.pixel_size_nm:.1f}) nm"
        self.readout.emit(f"{where}  value={value:.4g}")


def _bar_length_nm(image: DisplayImage, zoom: float) -> float:
    """The largest round length that fits in a fifth of the view.

    Round because a scale bar reading "137 nm" is one nobody can measure against
    by eye, which is the only thing a scale bar is for.
    """
    assert image.pixel_size_nm is not None
    width_px = image.size_px[0]
    quarter_nm = max(width_px * image.pixel_size_nm / 5, 1e-9)
    candidates = [length for length in BAR_LENGTHS_NM if length <= quarter_nm]
    return float(candidates[-1] if candidates else BAR_LENGTHS_NM[0])


def _to_qimage(rgb: np.ndarray) -> QImage:
    """An `(h, w, 3)` uint8 array as a `QImage`, copied.

    Copied on purpose: `QImage` does not own the buffer it is handed, and a view
    onto a numpy array that Python then frees is a crash that happens later,
    somewhere else.
    """
    height, width, _ = rgb.shape
    contiguous = np.ascontiguousarray(rgb)
    image = QImage(contiguous.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return image.copy()
