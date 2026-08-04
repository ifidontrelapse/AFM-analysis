"""AFM segmentation helpers (SAM2-based).

**Shim.** Split in M2-T07: the SAM2 runners went to
`nanoscope.infrastructure.models`, and `afm_to_rgb` / `overlay_masks` to
`nanoscope.infrastructure.imaging`, since neither has anything to do with SAM2.
Re-exports only; `src/pipeline.py` still imports the runners from here.

Deleted in M2-T15.
"""

from __future__ import annotations

from nanoscope.infrastructure.imaging import afm_to_rgb, overlay_masks
from nanoscope.infrastructure.models import run_sam2_from_blobs, run_sam2_from_boxes

__all__ = [
    "afm_to_rgb",
    "overlay_masks",
    "run_sam2_from_blobs",
    "run_sam2_from_boxes",
]
