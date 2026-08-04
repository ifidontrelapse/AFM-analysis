"""
Measurement helpers for particle heights and baselines.

**Shim.** The implementations moved to `nanoscope.core.science.measurement` in
M2-T06 — split into `height` (AFM) and `geometry` (any modality). This module
re-exports them and defines nothing; `src/pipeline.py` and `src/segmentation.py`
still import from here.

Deleted in M2-T15, once nothing imports `src`.
"""

from __future__ import annotations

from nanoscope.core.science.measurement import (
    create_circular_mask,
    get_clean_ring,
    measure_all_baseline,
    measure_geometry_from_mask,
    measure_height,
)

__all__ = [
    "create_circular_mask",
    "get_clean_ring",
    "measure_all_baseline",
    "measure_geometry_from_mask",
    "measure_height",
]
