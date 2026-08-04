"""
Utilities for loading AFM data from various formats.

**Shim.** The parsing moved to `nanoscope.core.science.io` and the file-reading to
`nanoscope.infrastructure.storage` in M2-T04; this module re-exports and defines
nothing. `src/preprocessing_pipeline.py` and `tests/unit/test_afm_io.py` still
import from here.

Deleted in M2-T15, once nothing imports `src`.
"""

from __future__ import annotations

from nanoscope.core.science.io import _read_nanoscope_z
from nanoscope.infrastructure.storage import (
    load_afm,
    load_microscopy_image,
)

__all__ = [
    "_read_nanoscope_z",
    "load_afm",
    "load_microscopy_image",
]
