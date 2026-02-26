"""Measurement helpers for particle heights and baselines."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def create_circular_mask(shape: tuple[int, int], cy: int, cx: int, radius: float) -> np.ndarray:
    """Create a boolean disk mask centered at (cy, cx)."""
    y, x = np.ogrid[: shape[0], : shape[1]]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def get_ring_baseline(
    z: np.ndarray,
    mask: np.ndarray,
    outer_px: int,
    inner_erode_px: int,
) -> float:
    """Median baseline in a ring around the particle mask."""
    outer = binary_dilation(mask, iterations=outer_px)
    inner = binary_erosion(mask, iterations=inner_erode_px)
    ring = outer & ~inner

    if ring.sum() == 0:
        return float(np.median(z))
    return float(np.median(z[ring]))


def measure_height(
    z: np.ndarray,
    mask: np.ndarray,
    outer_px: int = 7,
    inner_erode_px: int = 2,
) -> dict:
    """Compute height metrics relative to a local baseline."""
    baseline = get_ring_baseline(z, mask, outer_px, inner_erode_px)
    z_in_mask = z[mask]
    return {
        "height_nm": float(z_in_mask.max() - baseline),
        "mean_nm": float(z_in_mask.mean() - baseline),
        "baseline_nm": baseline,
        "area_px": int(mask.sum()),
    }
