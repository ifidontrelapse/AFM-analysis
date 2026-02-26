"""Particle detection utilities (seed generation)."""

from __future__ import annotations

import numpy as np
from skimage.feature import blob_log


def detect_seeds(z_above: np.ndarray, cfg: dict) -> np.ndarray:
    """Detect particle seeds using LoG on a normalized particles-only map."""
    px = cfg["pixel_size_nm"]

    min_sigma = cfg["particle_min_nm"] / (2 * np.sqrt(2) * px)
    max_sigma = cfg["particle_max_nm"] / (2 * np.sqrt(2) * px)

    z_norm = z_above / (z_above.max() + 1e-9)
    blobs = blob_log(
        z_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=15,
        threshold=cfg["log_threshold"],
        overlap=cfg["log_overlap"],
    )
    return blobs  # [y, x, sigma]
