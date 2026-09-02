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
from PySide6.QtCore import QLineF, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractGraphicsShapeItem,
    QCheckBox,
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
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
from nanoscope.core.entities import AnalysisRun, Annotation, AnnotationSource, Detection
from nanoscope.gui.pixmaps import to_pixmap, to_qimage
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: How the two kinds of hand work are told apart. `MANUAL` is a box somebody
#: drew; `FROM_DETECTION` is one they accepted from the machine — and ADR-0044
#: made that distinction load-bearing for training, because *a model trained on
#: its own output is confirming itself*. A screen that draws them alike undoes it
#: in the one place an operator would have noticed.
ANNOTATION_STYLES: dict[AnnotationSource, tuple[str, Qt.PenStyle]] = {
    AnnotationSource.MANUAL: (tokens.SUCCESS, Qt.PenStyle.SolidLine),
    AnnotationSource.FROM_DETECTION: (tokens.WARNING, Qt.PenStyle.DashLine),
}

#: How wide the overlay's outline is, in screen pixels. Cosmetic on purpose: a
#: pen measured in scene units turns a circle into a filled blob at 32x.
OVERLAY_WIDTH_PX = 1.5

#: How far a press may travel and still be a click rather than a drag. A mouse
#: moves under a finger, and without the tolerance half an operator's clicks
#: would be pans that selected nothing.
CLICK_SLOP_PX = 3

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

    #: A box the operator dragged, in scene (pixel) coordinates. Only while the
    #: drawing tool is on — the same gesture pans when it is off (ADR-0071).
    box_drawn = Signal(tuple)

    #: An outline the operator closed: `((x, y), …)`, at least three vertices.
    polygon_drawn = Signal(tuple)

    #: The annotation whose box was clicked, by position in the layer, or
    #: `None`. Annotations are drawn above the detections precisely so that a
    #: click reaches them first (ADR-0070 §1, ADR-0076 §5).
    annotation_picked = Signal(object)

    #: A line the operator dragged: `((x1, y1), (x2, y2))` in scene pixels.
    line_drawn = Signal(tuple)

    #: A stroke the operator painted, as a boolean mask the size of the scan.
    #: Emitted when the brush is lifted; the scan itself is never touched — the
    #: viewer shows the file (ADR-0056, ADR-0073 §4).
    mask_painted = Signal(object)

    #: The index of the overlay item that was clicked, or `None` for a click on
    #: bare image. A **click**, not a press: this view drags to pan (M5-T05), so
    #: a selection is a release that did not move (ADR-0065).
    picked = Signal(object)

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
        self._annotations: list[QGraphicsItem] = []
        self._painted: list[QGraphicsItem] = []
        self._rulers: list[QGraphicsItem] = []
        self._pressed_at: QPoint | None = None
        self._drawing = False
        self._outlining = False
        self._vertices: list[QPointF] = []
        self._sketch: QGraphicsPathItem | None = None
        self._painting = False
        self._measuring = False
        self._brush_px = 8
        self._stroke: np.ndarray | None = None
        self._stroke_item: QGraphicsPixmapItem | None = None
        #: The shape being dragged right now — a rectangle for the box tool,
        #: a line for the ruler. One at a time, because one tool at a time.
        self._rubber: QGraphicsRectItem | QGraphicsLineItem | None = None
        self._origin: QPointF | None = None

    def show_pixmap(self, pixmap: QPixmap) -> None:
        self._item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.reset_zoom()

    def clear(self) -> None:
        self._item.setPixmap(QPixmap())
        self.setSceneRect(QRectF())
        self.draw_detections(())
        self.draw_masks(())
        self.draw_annotations(())
        self.draw_painted(())
        self.draw_rulers(())

    def draw_annotations(self, annotations: Iterable[Annotation]) -> None:
        """The hand work, **above** everything else in the scene.

        Above because it is what the operator is working on, and what a click
        should reach first when M7-T02's tools arrive.
        """
        for item in self._annotations:
            self.scene().removeItem(item)
        self._annotations = [_annotation_item(one) for one in annotations]
        for item in self._annotations:
            item.setZValue(1.0)
            self.scene().addItem(item)

    @property
    def annotation_overlay(self) -> list[QGraphicsItem]:
        return list(self._annotations)

    def draw_painted(self, masks: Iterable[np.ndarray]) -> None:
        """Painted annotations, outlined like every other mask (ADR-0064 §6)."""
        for item in self._painted:
            self.scene().removeItem(item)
        self._painted = [_outline(mask, tokens.SUCCESS) for mask in masks]
        for item in self._painted:
            item.setZValue(1.0)
            self.scene().addItem(item)

    @property
    def painted_overlay(self) -> list[QGraphicsItem]:
        return list(self._painted)

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

    @property
    def drawing(self) -> bool:
        return self._drawing

    @property
    def outlining(self) -> bool:
        return self._outlining

    def set_outlining(self, on: bool) -> None:
        """Turn the polygon tool on, and panning off with it (ADR-0071 §2)."""
        self._outlining = on
        self._discard_sketch()
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def add_vertex(self, point: QPointF) -> None:
        """One click of an outline, drawn as it grows.

        Visible while it is being made, because an outline the operator cannot
        see until it is finished is one they draw twice.
        """
        self._vertices.append(point)
        path = QPainterPath(self._vertices[0])
        for vertex in self._vertices[1:]:
            path.lineTo(vertex)

        if self._sketch is None:
            self._sketch = QGraphicsPathItem()
            pen = QPen(tokens.qcolor(tokens.SUCCESS))
            pen.setWidthF(OVERLAY_WIDTH_PX)
            pen.setCosmetic(True)
            self._sketch.setPen(pen)
            self._sketch.setZValue(2.0)
            self.scene().addItem(self._sketch)
        self._sketch.setPath(path)

    def close_outline(self) -> None:
        """Finish it, if it is one. Fewer than three vertices is not an outline."""
        vertices = [(vertex.x(), vertex.y()) for vertex in self._vertices]
        self._discard_sketch()
        if len(vertices) >= 3:
            self.polygon_drawn.emit(tuple(vertices))

    def _discard_sketch(self) -> None:
        if self._sketch is not None:
            self.scene().removeItem(self._sketch)
        self._sketch = None
        self._vertices = []

    @property
    def painting(self) -> bool:
        return self._painting

    @property
    def measuring(self) -> bool:
        return self._measuring

    def set_measuring(self, on: bool) -> None:
        """Turn the ruler on, and panning off with it (ADR-0071 §2)."""
        self._measuring = on
        self._discard_rubber()
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def draw_rulers(self, lines: Iterable[tuple[tuple[float, float], tuple[float, float]]]) -> None:
        """The stored lines, above the scan and below the hand-drawn shapes."""
        for item in self._rulers:
            self.scene().removeItem(item)
        self._rulers = [_ruler_item(start, end) for start, end in lines]
        for item in self._rulers:
            item.setZValue(1.0)
            self.scene().addItem(item)

    @property
    def ruler_overlay(self) -> list[QGraphicsItem]:
        return list(self._rulers)

    def _discard_rubber(self) -> None:
        if self._rubber is not None:
            self.scene().removeItem(self._rubber)
        self._rubber, self._origin = None, None

    def set_painting(self, on: bool, *, brush_px: int = 8) -> None:
        """Turn the brush on, and panning off with it (ADR-0071 §2)."""
        self._painting = on
        self._brush_px = max(1, brush_px)
        self._discard_stroke()
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def set_brush(self, brush_px: int) -> None:
        self._brush_px = max(1, brush_px)

    def paint_at(self, point: QPointF) -> None:
        """Add one dab to the stroke in progress, and show it.

        Into a mask of its own, never into the scan: a tool that edited the data
        an operator is measuring would be the worst version of this feature.
        """
        pixmap = self._item.pixmap()
        if pixmap.isNull():
            return
        if self._stroke is None:
            self._stroke = np.zeros((pixmap.height(), pixmap.width()), dtype=bool)

        height, width = self._stroke.shape
        y, x = round(point.y()), round(point.x())
        radius = self._brush_px
        ys, xs = np.ogrid[:height, :width]
        self._stroke |= (ys - y) ** 2 + (xs - x) ** 2 <= radius**2
        self._show_stroke()

    def finish_stroke(self) -> None:
        """Hand the stroke over, if anything was painted."""
        stroke = self._stroke
        self._discard_stroke()
        if stroke is not None and stroke.any():
            self.mask_painted.emit(stroke)

    def _show_stroke(self) -> None:
        if self._stroke is None:
            return
        rgb = np.zeros((*self._stroke.shape, 3), dtype=np.uint8)
        colour = tokens.qcolor(tokens.SUCCESS)
        rgb[self._stroke] = (colour.red(), colour.green(), colour.blue())
        image = to_qimage(rgb)
        image.setAlphaChannel(
            to_qimage(np.repeat((self._stroke * 160).astype(np.uint8)[..., None], 3, axis=2))
        )
        if self._stroke_item is None:
            self._stroke_item = QGraphicsPixmapItem()
            self._stroke_item.setZValue(2.0)
            self.scene().addItem(self._stroke_item)
        self._stroke_item.setPixmap(QPixmap.fromImage(image))

    def _discard_stroke(self) -> None:
        if self._stroke_item is not None:
            self.scene().removeItem(self._stroke_item)
        self._stroke_item = None
        self._stroke = None

    def set_drawing(self, on: bool) -> None:
        """Turn the box tool on, and panning off with it.

        A tool that draws *and* pans on the same gesture does the wrong one half
        the time, so the drag mode changes with the tool and the cursor says so.
        """
        self._drawing = on
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def highlight_annotation(self, index: int | None) -> None:
        """Thicken the selected box, and only that one."""
        for position, item in enumerate(self._annotations):
            pen = item.pen()  # type: ignore[attr-defined]  # every shape here has one
            pen.setWidthF(OVERLAY_WIDTH_PX * (3 if position == index else 1))
            item.setPen(pen)  # type: ignore[attr-defined]

    def highlight(self, index: int | None) -> None:
        """Thicken the selected outline, and only that one."""
        for position, item in enumerate(self._overlay):
            pen = item.pen()  # type: ignore[attr-defined]  # every shape here has one
            pen.setWidthF(OVERLAY_WIDTH_PX * (3 if position == index else 1))
            item.setPen(pen)  # type: ignore[attr-defined]

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

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt's name
        """Double-click closes an outline — the gesture every annotation tool uses."""
        if self._outlining:
            self.close_outline()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt's name
        if self._outlining and not self._item.pixmap().isNull():
            self.add_vertex(self.mapToScene(event.position().toPoint()))
            return
        if self._painting:
            self.paint_at(self.mapToScene(event.position().toPoint()))
            return
        if self._measuring and not self._item.pixmap().isNull():
            self._origin = self.mapToScene(event.position().toPoint())
            self._rubber = QGraphicsLineItem(QLineF(self._origin, self._origin))
            pen = QPen(tokens.qcolor(tokens.ACCENT))
            pen.setWidthF(OVERLAY_WIDTH_PX)
            pen.setCosmetic(True)
            self._rubber.setPen(pen)
            self._rubber.setZValue(2.0)
            self.scene().addItem(self._rubber)
            return
        self._pressed_at = event.position().toPoint()
        if self._drawing and not self._item.pixmap().isNull():
            self._origin = self.mapToScene(self._pressed_at)
            self._rubber = QGraphicsRectItem(QRectF(self._origin, self._origin))
            pen = QPen(tokens.qcolor(tokens.SUCCESS))
            pen.setWidthF(OVERLAY_WIDTH_PX)
            pen.setCosmetic(True)
            self._rubber.setPen(pen)
            self._rubber.setZValue(2.0)
            self.scene().addItem(self._rubber)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt's name
        """A release within a few pixels of its press is a click, not a drag.

        Three pixels because a mouse moves under a finger; without the
        tolerance, half of an operator's clicks would be pans that selected
        nothing.
        """
        if self._painting:
            self.finish_stroke()
            return
        if isinstance(self._rubber, QGraphicsLineItem) and self._origin is not None:
            line = self._rubber.line()
            self._discard_rubber()
            self._pressed_at = None
            self.line_drawn.emit(((line.x1(), line.y1()), (line.x2(), line.y2())))
            return

        if isinstance(self._rubber, QGraphicsRectItem) and self._origin is not None:
            rect = self._rubber.rect()
            self._discard_rubber()
            self._pressed_at = None
            self.box_drawn.emit((rect.left(), rect.top(), rect.right(), rect.bottom()))
            return

        super().mouseReleaseEvent(event)
        start = self._pressed_at
        self._pressed_at = None
        if start is None or (event.position().toPoint() - start).manhattanLength() > CLICK_SLOP_PX:
            return

        point = self.mapToScene(event.position().toPoint())
        for index, item in enumerate(self._annotations):
            if item.sceneBoundingRect().contains(point):
                self.annotation_picked.emit(index)
                return
        self.annotation_picked.emit(None)

        for index, item in enumerate(self._overlay):
            if item.sceneBoundingRect().contains(point):
                self.picked.emit(index)
                return
        self.picked.emit(None)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt's name
        if isinstance(self._rubber, QGraphicsLineItem) and self._origin is not None:
            self._rubber.setLine(QLineF(self._origin, self.mapToScene(event.position().toPoint())))
        elif isinstance(self._rubber, QGraphicsRectItem) and self._origin is not None:
            self._rubber.setRect(
                QRectF(self._origin, self.mapToScene(event.position().toPoint())).normalized()
            )
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
        session.annotations_changed.connect(lambda _annotations: self._draw_overlay())
        session.rulers_changed.connect(lambda _rulers: self._draw_overlay())

        self.view = ImageView(self)
        self.view.hovered.connect(self._describe)
        #: Both directions of M6's selection criterion, and neither of them is
        #: a widget talking to a widget: the canvas asks the session, and the
        #: session answers everyone (ADR-0065).
        self.view.picked.connect(session.select_particle)
        session.particle_selected.connect(self.view.highlight)
        self.view.annotation_picked.connect(self._annotation_picked)
        session.annotation_selected.connect(lambda _id: self._draw_overlay())

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

        self.show_annotations = QCheckBox("Annotations", self)
        self.show_annotations.setChecked(True)
        self.show_annotations.setToolTip(
            "Boxes a person drew or accepted. The one thing in a project that "
            "cannot be recomputed (ADR-0044)."
        )
        self.show_annotations.toggled.connect(lambda _: self._draw_overlay())

        #: Which array is on screen. ADR-0056's rule was never "show the file
        #: and nothing else" — it was *never show something the file does not
        #: contain without saying so*, and this label is how that promise
        #: survives M6-T01 having something else to show.
        self.stage_label = QLabel("", self)
        self.scale_label = QLabel("", self)
        #: Why the canvas is empty, beside the empty canvas. The reason already
        #: reached the status bar — one transient line, under a readout that
        #: overwrites it — and an operator whose folder of scans would not open
        #: read a blank viewer as "the application is broken" rather than as
        #: "this file has no reader" (2026-08-30).
        self.failure_label = QLabel("", self)
        self.failure_label.setWordWrap(True)
        session.failed.connect(self._show_failure)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Colormap", self))
        controls.addWidget(self.colormap)
        controls.addWidget(self.full_range)
        controls.addWidget(self.show_detections)
        controls.addWidget(self.show_masks)
        controls.addWidget(self.show_annotations)
        #: Beside the controls rather than at the far right: it was clipped to
        #: "result (flatte…" when it competed with the scale bar for the end of
        #: the row, and a label nobody can finish reading is not a statement.
        controls.addWidget(self.stage_label)
        #: Where the stage and the scale bar are not, because it only ever has
        #: text when there is no image and those two are empty.
        controls.addWidget(self.failure_label)
        controls.addStretch(1)
        controls.addWidget(self.scale_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.view)

        self.show_image(session.image)
        self._draw_overlay()

    def _show_failure(self, message: str) -> None:
        """Say why there is nothing to look at — and only then.

        `failed` carries every refusal the session makes, including exports and
        removals, and those have nothing to do with the canvas. The guard is
        that this panel speaks only when it has nothing to draw.
        """
        if self._image is None:
            self.failure_label.setText(message)

    def _annotation_picked(self, index: int | None) -> None:
        """A position in the layer becomes the id the session speaks in."""
        annotations = self._session.annotations
        chosen = annotations[index].id if index is not None and index < len(annotations) else None
        self._session.select_annotation(chosen)

    def _run_changed(self, _run: AnalysisRun | None) -> None:
        self._draw_overlay()

    def _draw_overlay(self) -> None:
        """Draw the current run's detections, or none of them."""
        run = self._session.run
        detections = run.detections if run is not None and self.show_detections.isChecked() else ()
        self.view.draw_detections(detections)
        self.view.highlight(self._session.particle)
        annotations = self._session.annotations if self.show_annotations.isChecked() else ()
        self.view.draw_annotations(annotations)
        #: A painted mask is a file, so the layer asks for it rather than
        #: carrying it — and what is drawn is what was painted, not its
        #: bounding box (ADR-0072's rule, third shape).
        self.view.draw_painted(
            [
                painted
                for annotation in annotations
                if (painted := self._session.mask_of(annotation)) is not None
            ]
        )
        self.show_annotations.setText(_annotation_count(self._session.annotations))
        selected = self._session.selected_annotation
        self.view.highlight_annotation(
            next(
                (position for position, one in enumerate(annotations) if one.id == selected),
                None,
            )
        )
        self.view.draw_rulers(
            [(ruler.start, ruler.end) for ruler in self._session.rulers]
            if self.show_annotations.isChecked()
            else []
        )

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

        #: An image arrived, so whatever the last refusal was, it is not what is
        #: on screen any more.
        self.failure_label.setText("")

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
        self.view.show_pixmap(to_pixmap(rgb))
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


def _outline(mask: np.ndarray, colour: str = tokens.SUCCESS) -> QGraphicsItem:
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
    pen = QPen(tokens.qcolor(colour))
    pen.setWidthF(OVERLAY_WIDTH_PX)
    pen.setCosmetic(True)
    item.setPen(pen)
    item.setBrush(QBrush())
    return item


def _annotation_item(annotation: Annotation) -> QGraphicsItem:
    """One box a person drew, with the label that is the reason it exists.

    The label does **not** scale with the zoom: one that grows to fill the
    screen at 32x is a label nobody can read at 32x.
    """
    colour, style = ANNOTATION_STYLES[annotation.source]
    x1, y1, x2, y2 = annotation.box
    item: QAbstractGraphicsShapeItem
    if annotation.points is None:
        item = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
    else:
        #: The outline, not its bounding box: a polygon drawn as a rectangle is
        #: a shape nobody made (ADR-0072).
        item = QGraphicsPolygonItem(QPolygonF([QPointF(x, y) for x, y in annotation.points]))

    pen = QPen(tokens.qcolor(colour))
    pen.setWidthF(OVERLAY_WIDTH_PX)
    pen.setCosmetic(True)
    pen.setStyle(style)
    item.setPen(pen)
    item.setBrush(QBrush())
    item.setToolTip(f"{annotation.label} ({annotation.source})")

    label = QGraphicsSimpleTextItem(annotation.label, item)
    label.setBrush(QBrush(tokens.qcolor(colour)))
    label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
    label.setPos(x1, y1)
    return item


def _annotation_count(annotations: tuple[Annotation, ...]) -> str:
    """The toggle's label: how much hand work this scan carries."""
    return "Annotations" if not annotations else f"Annotations ({len(annotations)})"


def _ruler_item(start: tuple[float, float], end: tuple[float, float]) -> QGraphicsItem:
    """One measured line. The number is in the panel; this is where it was."""
    item = QGraphicsLineItem(QLineF(QPointF(*start), QPointF(*end)))
    pen = QPen(tokens.qcolor(tokens.ACCENT))
    pen.setWidthF(OVERLAY_WIDTH_PX)
    pen.setCosmetic(True)
    item.setPen(pen)
    return item
