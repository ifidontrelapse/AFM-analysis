"""Model-backed adapters: YOLO detection and SAM2 segmentation.

Everything here needs weights, a framework, or both. That is the whole reason the
layer exists — `core` is defined by not importing them, and M2-T09 makes that a
test rather than a convention.

Heavy imports are function-local throughout. CI installs none of torch,
ultralytics, sam2 or patched_yolo_infer (M1-T08), and importing this package must
stay possible without them.
"""

from nanoscope.infrastructure.models.sam2 import run_sam2_from_blobs, run_sam2_from_boxes
from nanoscope.infrastructure.models.yolo import YoloDetector

__all__ = ["YoloDetector", "run_sam2_from_blobs", "run_sam2_from_boxes"]
