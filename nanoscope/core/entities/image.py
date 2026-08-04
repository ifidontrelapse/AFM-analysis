"""Image records: what a file becomes, and what preprocessing makes of it.

Moved verbatim from `src/types.py` in M2-T02. Field names and order are part of
the characterization golden — it records `dataclasses.fields(...)` — so this is a
copy, not a tidy-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class AFMRawData:
    """Raw output of load_afm — before any preprocessing."""

    z_raw: np.ndarray
    pixel_size_nm: float
    scan_size_nm: float


@dataclass
class PreprocessingResult:
    z_raw: np.ndarray  # raw Z-map straight from the file
    z_flat: np.ndarray  # after plane + line flattening
    z_result: np.ndarray  # z_flat - substrate (particles above substrate)
    substrate: np.ndarray  # estimated substrate surface
    pixel_size_nm: float  # nm/pixel
    scan_size_nm: float  # full scan size in nm
    # `estimate_radius_otsu` returns floats, ints and ndarrays under string keys,
    # so `Any` is the honest value type, not a placeholder. Was a bare `dict`
    # before the move; `disallow_any_generics` is on for `nanoscope.*`, and an
    # annotation is not a number — the golden records field names, not types.
    sizes: dict[str, Any]  # output of estimate_radius_otsu
    opening_radius: int  # morphological opening radius used


@dataclass
class MicroscopyData:
    """
    Image data for SEM or TEM — no preprocessing, no height map.
    Geometry (area, radius, circularity) is derived from segmentation masks.
    """

    image: np.ndarray
    nm_per_pixel: float | None  # None if physical scale is unknown
    modality: Literal["sem", "tem"]
