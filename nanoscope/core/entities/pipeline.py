"""What the pipeline is asked for, and what it returns.

Moved verbatim from `src/types.py` in M2-T02. The golden records the sorted field
names of both classes, so adding or renaming one here is drift — including a field
that looks obviously missing.

`measurements: pd.DataFrame` puts pandas in the domain layer. That is today's
design, not an endorsement; revisit it with the import-weight work in M2-T09.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from nanoscope.core.entities.detection import Detection

if TYPE_CHECKING:
    # Annotation-only. `from __future__ import annotations` means the string
    # "pd.DataFrame" is never evaluated, so importing pandas at run time bought
    # nothing but weight: it was ~380 of the modules that `import
    # nanoscope.core.entities` used to load (M2-T09).
    #
    # This does not change `dataclasses.fields(...).type`, which is that same
    # string either way, and the golden records field *names*. It would break
    # `typing.get_type_hints(PipelineResult)` at run time; nothing calls it.
    import pandas as pd


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
    # False since ADR-0021 (B7): `_prepare_image` produces exactly one
    # `yolo_size` square and the crop shape is the same square, so the tiled
    # backend has always produced a single crop — the direct backend's work,
    # more slowly. The backend is kept for the day the input is larger.
    yolo_use_tiling: bool = False
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
    # `dict[str, Any]`: a mask entry mixes ndarrays with floats under string
    # keys (`src/segmentation.py`). Was `list[dict]` before the move.
    masks: list[dict[str, Any]]  # empty if mode="detect"
    measurements: pd.DataFrame  # empty if mode="detect"
    pixel_size_nm: float | None  # None if nm_per_pixel was unknown
    detector_name: str
    mode: str
    modality: str  # "afm" | "sem" | "tem"
