"""Shape metrics from a binary mask — area, radius, circularity, aspect ratio.

Moved verbatim from `src/measure.py` in M2-T06, and this is the split the task
existed for: none of it needs a height map, so it is the measurement path SEM and
TEM actually use. It was trapped inside an AFM module, which is why
`src/segmentation.py` reaches into `src.measure` for it.

`nm_per_pixel=None` is a supported answer, not an error — an image whose physical
scale is unknown still has pixel-space geometry, and the `_nm` keys come back None.
`PixelScale` (M2-T02) is the eventual home for that arithmetic.
"""

from __future__ import annotations

import numpy as np


def measure_geometry_from_mask(
    mask: np.ndarray,
    nm_per_pixel: float | None,
) -> dict:
    """
    Compute geometric properties of a particle from its binary mask.
    Used for SEM/TEM where no height map is available.

    Args:
        mask:          boolean mask of a single particle
        nm_per_pixel:  physical scale; if None, nm values are returned as None

    Returns:
        dict with keys:
            area_px, area_nm2,
            radius_px, radius_nm,
            circularity,       (4π·area / perimeter², 1.0 = perfect circle)
            aspect_ratio       (major_axis / minor_axis)
    """
    from skimage.measure import label, regionprops

    props = regionprops(label(mask))
    if not props:
        return {
            "area_px": 0,
            "area_nm2": None,
            "radius_px": 0.0,
            "radius_nm": None,
            "circularity": None,
            "aspect_ratio": None,
        }

    p = props[0]
    area_px = int(p.area)
    perimeter = float(p.perimeter) if p.perimeter > 0 else 1.0
    radius_px = float(p.equivalent_diameter_area / 2)
    circularity = float(4 * np.pi * area_px / perimeter**2)
    aspect_ratio = float(
        p.major_axis_length / p.minor_axis_length if p.minor_axis_length > 0 else 1.0
    )

    if nm_per_pixel is not None:
        area_nm2 = area_px * nm_per_pixel**2
        radius_nm = radius_px * nm_per_pixel
    else:
        area_nm2 = None
        radius_nm = None

    return {
        "area_px": area_px,
        "area_nm2": area_nm2,
        "radius_px": radius_px,
        "radius_nm": radius_nm,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
    }
