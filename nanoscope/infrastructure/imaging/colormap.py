"""Turning arrays into pictures — colormaps and mask overlays.

Moved verbatim from `src/segmentation.py` in M2-T07. Neither function has anything
to do with SAM2: `afm_to_rgb` applies a matplotlib colormap and `overlay_masks`
blends colours over an RGB image. They were in a segmentation module because that
is where they were first needed, which is the same accident M2-T06 untangled in
`measure.py`.

`infrastructure`, not `core`: matplotlib is a rendering dependency, and the domain
is defined by not having one.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def afm_to_rgb(
    z: np.ndarray, colormap: str = "afmhot", clip_percentile: float = 99.0
) -> np.ndarray:
    """Convert a Z-map (float) into uint8 RGB for SAM input."""
    lo, hi = z.min(), np.percentile(z, clip_percentile)
    z_clip = np.clip(z, lo, hi)
    z_norm = (z_clip - lo) / (hi - lo + 1e-9)
    cmap = plt.get_cmap(colormap)
    rgb = (cmap(z_norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def overlay_masks(rgb_img: np.ndarray, sam_results: list[dict], alpha: float = 0.45) -> np.ndarray:
    """Overlay SAM masks with random colors for visualization."""
    overlay = rgb_img.copy().astype(float)
    rng = np.random.default_rng(0)
    for r in sam_results:
        color = rng.integers(80, 255, 3).astype(float)
        for c in range(3):
            overlay[:, :, c][r["mask"]] = (
                alpha * color[c] + (1 - alpha) * overlay[:, :, c][r["mask"]]
            )
    return overlay.clip(0, 255).astype(np.uint8)
