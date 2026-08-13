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

from collections.abc import Iterable

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.settings import COLORMAP_SETTING
from nanoscope.application.use_cases.display import (
    COLORMAPS,
    STAGE_LABELS,
    DisplayImage,
    render,
    stage_image,
    value_range,
)
from nanoscope.core.entities import AnalysisRun, Detection
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: How wide the overlay's outline is, in screen pixels. Cosmetic on purpose: a
#: pen measured in scene units turns a circle into a filled blob at 32x.
OVERLAY_WIDTH_PX = 1.5

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
        self._overlay: list[QGraphicsItem] = []
        self._masks: list[QGraphicsItem] = []

    def show_pixmap(self, pixmap: QPixmap) -> None:
        self._item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.reset_zoom()

    def clear(self) -> None:
        self._item.setPixmap(QPixmap())
        self.setSceneRect(QRectF())
        self.draw_detections(())
        self.draw_masks(())

    def draw_masks(self, masks: Iterable[np.ndarray]) -> None:
        """Outline each mask, in scene coordinates like everything else.

        An **outline**, not a filled sheet: a filled overlay hides the pixels it
        describes, and those pixels are the measurement (ADR-0064).
        """
        for item in self._masks:
            self.scene().removeItem(item)
        self._masks = [_outline(mask) for mask in masks]
        for item in self._masks:
            self.scene().addItem(item)

    @property
    def mask_overlay(self) -> list[QGraphicsItem]:
        return list(self._masks)

    def draw_detections(self, detections: Iterable[Detection]) -> None:
        """Put one item on each particle, in **scene** coordinates.

        The view transforms the scene, so an item placed at `(x_px, y_px)` stays
        on its particle at every zoom and pan for nothing. Painting over the
        viewport instead would mean redoing that arithmetic, and being wrong the
        first time somebody drags.
        """
        for item in self._overlay:
            self.scene().removeItem(item)
        self._overlay = [_shape(detection) for detection in detections]
        for item in self._overlay:
            self.scene().addItem(item)

    @property
    def overlay(self) -> list[QGraphicsItem]:
        return list(self._overlay)

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
        session.preview_changed.connect(lambda _preview: self.show_image(session.image))
        session.run_changed.connect(self._run_changed)

        self.view = ImageView(self)
        self.view.hovered.connect(self._describe)

        self.colormap = QComboBox(self)
        self.colormap.addItems(COLORMAPS)
        self.colormap.currentTextChanged.connect(lambda _: self._redraw())
        #: The combo is *this scan*; the stored preference is *the default*
        #: (M5-T09). Two controls writing one key would fight; one reads it.
        self._session = session
        session.settings_changed.connect(self._apply_default_colormap)
        self._apply_default_colormap()

        self.full_range = QCheckBox("Full range", self)
        self.full_range.setToolTip(
            "Off: the 2nd-98th percentile, so one hot pixel does not flatten the image.\n"
            "On: every value, so nothing is clipped."
        )
        self.full_range.toggled.connect(lambda _: self._redraw())

        self.show_detections = QCheckBox("Detections", self)
        self.show_detections.setChecked(True)
        self.show_detections.setToolTip(
            "The overlay covers the data it describes; turning it off is how "
            "'what does this look like without the circles?' gets answered."
        )
        self.show_detections.toggled.connect(lambda _: self._draw_overlay())

        self.show_masks = QCheckBox("Masks", self)
        self.show_masks.setChecked(True)
        self.show_masks.setToolTip(
            "Segmentation outlines from the run in this session. Masks are not "
            "stored, so a run read back from the project has none (ADR-0042)."
        )
        self.show_masks.toggled.connect(lambda _: self._draw_overlay())

        #: Which array is on screen. ADR-0056's rule was never "show the file
        #: and nothing else" — it was *never show something the file does not
        #: contain without saying so*, and this label is how that promise
        #: survives M6-T01 having something else to show.
        self.stage_label = QLabel("", self)
        self.scale_label = QLabel("", self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Colormap", self))
        controls.addWidget(self.colormap)
        controls.addWidget(self.full_range)
        controls.addWidget(self.show_detections)
        controls.addWidget(self.show_masks)
        #: Beside the controls rather than at the far right: it was clipped to
        #: "result (flatte…" when it competed with the scale bar for the end of
        #: the row, and a label nobody can finish reading is not a statement.
        controls.addWidget(self.stage_label)
        controls.addStretch(1)
        controls.addWidget(self.scale_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.view)

        self.show_image(session.image)
        self._draw_overlay()

    def _run_changed(self, _run: AnalysisRun | None) -> None:
        self._draw_overlay()

    def _draw_overlay(self) -> None:
        """Draw the current run's detections, or none of them."""
        run = self._session.run
        detections = run.detections if run is not None and self.show_detections.isChecked() else ()
        self.view.draw_detections(detections)
        masks = _mask_arrays(run) if run is not None and self.show_masks.isChecked() else ()
        self.view.draw_masks(masks)
        #: Hidden entirely when there are none: a control for something that
        #: does not exist teaches an operator to ignore the row it sits in.
        self.show_masks.setVisible(bool(run is not None and run.masks))
        #: The count rides on the checkbox rather than on a label of its own:
        #: the row was already carrying a colormap, a range, a stage and a scale
        #: bar, and the sixth widget was **clipped mid-word** in a real
        #: window. "Detections (0)" and an unticked box are also two different
        #: statements, which a separate label had to spell out.
        self.show_detections.setText(_count(run))

    def _apply_default_colormap(self) -> None:
        stored = str(self._session.preference(COLORMAP_SETTING, COLORMAPS[0]))
        if stored in COLORMAPS:
            #: `setCurrentText` with the current text emits nothing, so this
            #: cannot loop back through `_redraw`.
            self.colormap.setCurrentText(stored)

    # ── What it shows ─────────────────────────────────────────────────────────

    def show_image(self, image: DisplayImage | None) -> None:
        """Draw what the session loaded, or empty the canvas.

        The loading happened in the viewmodel — a widget that reads a file
        decides *what* to read, which is the line Architecture §2.3 draws. What
        is left here is the part that is genuinely presentation: an array, a
        colormap, and a pixmap.
        """
        stage = self._session.stage
        self._image = None if image is None else stage_image(stage, image, self._session.preview)
        if self._image is None:
            self.view.clear()
            self.scale_label.setText("")
            self.stage_label.setText("")
            return

        #: The stage alone, not "showing: raw" — six widgets share this row and
        #: the words that carry no information are the ones that clip the ones
        #: that do. The long form stays as the tooltip.
        self.stage_label.setText(str(stage))
        self._draw_overlay()
        self.stage_label.setToolTip(STAGE_LABELS[stage])
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


def _shape(detection: Detection) -> QGraphicsItem:
    """A box when the detector produced one, a circle when it did not.

    `bbox` is `None` on the blob path (ADR-0031), and drawing an invented box
    around a circle would be a shape nothing found — the same substitution
    ADR-0028 removed from `confidence`.
    """
    pen = QPen(tokens.qcolor(tokens.ACCENT))
    pen.setWidthF(OVERLAY_WIDTH_PX)
    #: Cosmetic: the width is in screen pixels, so the outline stays one line
    #: thick at 32x instead of swallowing the particle it marks.
    pen.setCosmetic(True)

    if detection.bbox is not None:
        x1, y1, x2, y2 = detection.bbox
        item: QGraphicsItem = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
    else:
        radius = detection.radius_px
        item = QGraphicsEllipseItem(
            QRectF(detection.x_px - radius, detection.y_px - radius, 2 * radius, 2 * radius)
        )
    item.setPen(pen)  # type: ignore[attr-defined]  # both shapes have one
    item.setBrush(QBrush())  # type: ignore[attr-defined]  # outline only: the data is underneath
    return item


def _count(run: AnalysisRun | None) -> str:
    """The checkbox's label: what there is to show, or that there is nothing."""
    return "Detections" if run is None else f"Detections ({len(run.detections)})"


def _mask_arrays(run: AnalysisRun) -> list[np.ndarray]:
    """The boolean arrays out of the mask entries, and nothing else from them.

    A mask entry also carries scores and geometry; the viewer wants the shape.
    """
    return [entry["mask"] for entry in run.masks if entry.get("mask") is not None]


def _outline(mask: np.ndarray) -> QGraphicsItem:
    """One mask as a path around its pixels.

    Built from the mask's own rectangles rather than a contour finder: tracing
    contours is `skimage`, which `gui/` may not import (Architecture §3.2), and
    a per-row span is exact where a polygon approximation is a second shape.
    """
    path = QPainterPath()
    rows = np.asarray(mask, dtype=bool)
    for y, row in enumerate(rows):
        starts = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))) == 1)
        ends = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))) == -1)
        for x1, x2 in zip(starts, ends, strict=True):
            path.addRect(QRectF(float(x1), float(y), float(x2 - x1), 1.0))

    item = QGraphicsPathItem(path.simplified())
    pen = QPen(tokens.qcolor(tokens.SUCCESS))
    pen.setWidthF(OVERLAY_WIDTH_PX)
    pen.setCosmetic(True)
    item.setPen(pen)
    item.setBrush(QBrush())
    return item
