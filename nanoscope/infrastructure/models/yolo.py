"""YOLOv8 particle detector — an adapter, because it needs model weights.

Moved verbatim from `src/detection/yolo_detector.py` in M2-T07. It subclasses
`BaseDetector` from `core.science.detection`, which is infrastructure depending on
the domain: the correct direction, and the one M2-T09's import-graph test enforces.

**Every heavy import here is function-local, and must stay that way.** CI installs
no torch, ultralytics, sam2 or patched_yolo_infer (M1-T08), so a module-level
import would turn the job red — which is the good outcome. Do not "fix" that with
a `try: import torch`; the point is that the domain can be tested without a GPU
stack, and a swallowed ImportError hides the day that stops being true.

Only `_prepare_image` is covered by the characterization golden, recorded as
`yolo_input_preparation` for the 7 phantoms that carry it. Inference itself is
outside the gate by PROJECT_RULES §6 — it is not reproducible enough to be a
baseline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from nanoscope.core.entities import Detection
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.science.detection import BaseDetector
from nanoscope.core.validation import ensure_height_map, ensure_positive
from nanoscope.core.values import Polarity

logger = logging.getLogger(__name__)


class YoloDetector(BaseDetector):
    """
    YOLOv8 particle detector.

    Two inference backends, selected at construction time via `use_tiling`:

    use_tiling=False (default since ADR-0021):
        Direct ultralytics YOLO inference on a single prepared image.

    use_tiling=True:
        patched_yolo_infer sliding-window approach (MakeCropsDetectThem +
        CombineDetections). Intended for dense particle fields, to avoid
        missed detections at tile boundaries — but see `_crop_steps`: with the
        current `_prepare_image` the window covers the whole image in one
        tile, so this backend does the direct backend's work more slowly. It
        is kept, not deleted, because the fix is to feed it a larger input
        (M3-T15 has to measure whether that is worth the inference cost).

    Both backends return the same list[Detection] in the coordinate space
    of the original z_above array.
    """

    def __init__(
        self,
        model_path: str = "./checkpoints/best12x.pt",
        use_tiling: bool = False,
        yolo_size: int = 640,
        # tiling-specific
        overlap_x: int = 25,
        overlap_y: int = 25,
        nms_threshold: float = 0.25,
        # shared
        conf: float = 0.5,
        iou: float = 0.7,
        polarity: Polarity = Polarity.BRIGHT_ON_DARK,
    ):
        self.model_path = model_path
        self.use_tiling = use_tiling
        self.yolo_size = yolo_size
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.nms_threshold = nms_threshold
        self.conf = conf
        self.iou = iou
        self.polarity = polarity
        # `Any` because the two possible types — `CombineDetections` and
        # ultralytics `Results` — live in heavy optional dependencies that
        # must not be imported at module level (see this module's docstring).
        # Without it, every attribute read off this field is a mypy error on
        # `None`; M3-T05 would otherwise have added a third.
        self._last_result: Any = None  # CombineDetections or ultralytics Results

    def _letterbox(self, original_shape: tuple[int, int]) -> tuple[float, int, int]:
        """Geometry of the map from `original_shape` into the model square.

        Returns `(scale, pad_x, pad_y)`: one scale for both axes, and the border
        that centres the scaled image in a `yolo_size` square. `_prepare_image`
        applies this map and `_scale_boxes` inverts it; they share the function
        so that they cannot disagree about it (ADR-0016).
        """
        h, w = original_shape
        scale = self.yolo_size / max(h, w)
        return (
            scale,
            (self.yolo_size - round(w * scale)) // 2,
            (self.yolo_size - round(h * scale)) // 2,
        )

    def _prepare_image(self, z_above: np.ndarray) -> np.ndarray:
        """AFM height-map → uint8 RGB (yolo_size × yolo_size).

        Two orderings in here are load-bearing, and both were defects:

        - **Normalise, then cast** (ADR-0015 / D-03). Casting a height map in
          nanometres to `uint8` first keeps only the integers inside its range
          and wraps the rest.
        - **Scale isotropically, then pad** (ADR-0016 / D-21). Resizing to a
          square turned a circular particle into an ellipse on any non-square
          scan. The padding goes on *after* the normalisation, so the border
          does not take part in the min-max stretch.

        The inversion is conditional on polarity (ADR-0023): the model wants dark
        particles, and a TEM image already has them.
        """
        import cv2

        scale, pad_x, pad_y = self._letterbox(z_above.shape)
        h, w = z_above.shape
        img = cv2.resize(z_above, (round(w * scale), round(h * scale)))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self.polarity is Polarity.BRIGHT_ON_DARK:
            # The weights were trained on inverted AFM height maps, so the model
            # looks for *dark* particles. Inverting a TEM image, where they are
            # already dark, hands it the background (D-12, ADR-0023).
            img = cv2.bitwise_not(img)
        # 255 is what a minimum height looks like after the inversion, so the
        # border reads as more substrate rather than as an edge (ADR-0016).
        img = cv2.copyMakeBorder(
            img,
            pad_y,
            self.yolo_size - img.shape[0] - pad_y,
            pad_x,
            self.yolo_size - img.shape[1] - pad_x,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def _scale_boxes(self, boxes: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Map boxes from model space back to original image coordinates.

        The exact inverse of `_prepare_image`'s letterbox: undo the border, then
        the one scale factor. One factor for both axes is the point — the old
        two-factor version stretched every box by the scan's aspect ratio.
        """
        scale, pad_x, pad_y = self._letterbox(original_shape)
        return (boxes - np.array([pad_x, pad_y, pad_x, pad_y])) / scale

    def _crop_steps(self, side: int) -> int:
        """How many crops the sliding window makes along an axis of `side` px.

        `get_crops_xy` computes `int((side - shape) / (shape * (1 - overlap))) + 1`.
        With `side == shape` that is 1, whatever the overlap — the finding behind
        ADR-0021, and the reason this is a method rather than a comment.
        """
        step = self.yolo_size * (1 - self.overlap_x / 100)
        return int((side - self.yolo_size) / step) + 1

    def _warn_if_single_crop(self, img: np.ndarray) -> bool:
        """Warn when the sliding window degenerates to one tile, and say so.

        Separate from `_detect_tiled` so it is testable without weights: the
        gate installs no ultralytics, and inference is outside it (§6).
        """
        if self._crop_steps(img.shape[1]) * self._crop_steps(img.shape[0]) > 1:
            return False
        logger.warning(
            "tiling requested but the %dx%d input is one %d px crop: this runs the "
            "direct backend's work more slowly and resolves nothing extra (ADR-0021)",
            img.shape[1],
            img.shape[0],
            self.yolo_size,
        )
        return True

    def _detect_tiled(
        self, img: np.ndarray, original_shape: tuple, pixel_size_nm: float | None
    ) -> list[Detection]:
        from patched_yolo_infer import CombineDetections, MakeCropsDetectThem

        self._warn_if_single_crop(img)

        crops = MakeCropsDetectThem(
            image=img,
            model_path=self.model_path,
            segment=False,
            shape_x=self.yolo_size,
            shape_y=self.yolo_size,
            overlap_x=self.overlap_x,
            overlap_y=self.overlap_y,
            conf=self.conf,
            iou=self.iou,
        )
        self._last_result = CombineDetections(crops, nms_threshold=self.nms_threshold)
        boxes = self._scale_boxes(
            np.array(self._last_result.filtered_boxes, dtype=np.float32),
            original_shape,
        )
        # After NMS across crops, so these are the scores of the boxes that
        # survived — the same list, in the same order (D-09, ADR-0028).
        confidences = np.array(self._last_result.filtered_confidences, dtype=np.float32)
        return self._boxes_to_detections(boxes, pixel_size_nm, confidences)

    def _detect_direct(
        self, img: np.ndarray, original_shape: tuple, pixel_size_nm: float | None
    ) -> list[Detection]:
        from ultralytics import YOLO

        model = YOLO(self.model_path)
        results = model(img, conf=self.conf, iou=self.iou)
        self._last_result = results

        boxes_yolo = results[0].boxes.xyxy.cpu().numpy()
        boxes = self._scale_boxes(boxes_yolo, original_shape)
        # `boxes.conf` is per box and already filtered by `conf=self.conf`, so it
        # aligns with `boxes.xyxy` row for row (D-09, ADR-0028).
        confidences = results[0].boxes.conf.cpu().numpy()
        return self._boxes_to_detections(boxes, pixel_size_nm, confidences)

    @staticmethod
    def _boxes_to_detections(
        boxes: np.ndarray, pixel_size_nm: float | None, confidences: np.ndarray | None = None
    ) -> list[Detection]:
        """Convert scaled boxes into entities, carrying each box's own score.

        Args:
            boxes: `(N, 4)` array of `(x1, y1, x2, y2)` in image pixels.
            pixel_size_nm: nm/pixel, or `None` when the scale is unknown.
            confidences: `(N,)` scores from the model, in the same order as
                `boxes`. `None` only for a caller that has none — the two
                backends always have them (D-09, ADR-0028).

        Returns:
            One `Detection` per box.

        Raises:
            InvalidParameterError: if `confidences` is given and its length does
                not match `boxes`. Zipping them would silently drop the tail, and
                a score attached to the wrong box is worse than no score.
        """
        if confidences is not None and len(confidences) != len(boxes):
            raise InvalidParameterError(
                f"got {len(confidences)} confidences for {len(boxes)} boxes; "
                "they must correspond one to one"
            )

        detections = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            radius_px = min(x2 - x1, y2 - y1) / 2
            detections.append(
                Detection(
                    x_px=cx,
                    y_px=cy,
                    radius_px=radius_px,
                    # No scale, no nanometres — never a pixel count in disguise
                    # (D-07, ADR-0019).
                    radius_nm=None if pixel_size_nm is None else radius_px * pixel_size_nm,
                    confidence=None if confidences is None else float(confidences[i]),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        return detections

    def detect(self, z_above: np.ndarray, pixel_size_nm: float | None) -> list[Detection]:
        # Before the weights are asked for anything: preparation resizes,
        # normalises and casts, and every one of those has its own opinion about
        # a 3-D array (ADR-0030).
        z_above = ensure_height_map(z_above, "z_above")
        ensure_positive(pixel_size_nm, "pixel_size_nm", allow_none=True)

        img = self._prepare_image(z_above)
        if self.use_tiling:
            return self._detect_tiled(img, z_above.shape, pixel_size_nm)
        return self._detect_direct(img, z_above.shape, pixel_size_nm)

    @property
    def last_result(self):
        """Raw result object from the last detect() call (for visualize_results)."""
        return self._last_result
