"""SAM 2 integration helpers.

These functions are intentionally isolated so the rest of the pipeline can
run without SAM installed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def afm_to_rgb(z: np.ndarray, colormap: str = "afmhot", clip_percentile: float = 99.0) -> np.ndarray:
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
