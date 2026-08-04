"""What the pipeline is asked for, and what it returns.

Moved verbatim from `src/types.py` in M2-T02. The golden records the sorted field
names of both classes, so adding or renaming one here is drift — including a field
that looks obviously missing.

`measurements: pd.DataFrame` puts pandas in the domain layer. That is today's
design, not an endorsement; revisit it with the import-weight work in M2-T09.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from nanoscope.core.entities.detection import Detection


@dataclass
class PipelineConfig:
    detector: Literal["log", "yolo"] = "log"
    mode: Literal["detect", "baseline", "segment"] = "segment"

    # LogDetector params
    log_overlap: float = 0.3
    log_percentile: float = 20.0
    log_threshold: float | None = None

    # YoloDetector params
    yolo_model_path: str = "./checkpoints/best12x.pt"
    yolo_use_tiling: bool = True
    yolo_conf: float = 0.5

    # SAM2 params
    sam2_outer_ring_px: int = 5
    sam2_inner_erode_px: int = 2

    # Measure params
    measure_outer_px: int = 5
    measure_inner_erode_px: int = 3


@dataclass
class PipelineResult:
    detections: list[Detection]
    masks: list[dict]  # empty if mode="detect"
    measurements: pd.DataFrame  # empty if mode="detect"
    pixel_size_nm: float | None  # None if nm_per_pixel was unknown
    detector_name: str
    mode: str
    modality: str  # "afm" | "sem" | "tem"
