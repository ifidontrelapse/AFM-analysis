"""
AFM preprocessing: plane levelling, trend removal, particles separated from background.

**Shim.** The implementations moved to `nanoscope.core.science.preprocessing` in
M2-T03; this module re-exports them and defines nothing. `src/preprocessing_pipeline.py`
and the characterization harness still import from here.

Deleted in M2-T15, once nothing imports `src`.
"""

from __future__ import annotations

from nanoscope.core.science.preprocessing import (
    build_substrate_map,
    estimate_radius_otsu,
    estimate_rough_radius,
    flatten_lines,
    flatten_plane,
    get_substrate_map,
)

__all__ = [
    "build_substrate_map",
    "estimate_radius_otsu",
    "estimate_rough_radius",
    "flatten_lines",
    "flatten_plane",
    "get_substrate_map",
]
