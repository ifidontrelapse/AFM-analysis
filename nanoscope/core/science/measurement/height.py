"""AFM height measurement: how far a particle stands above its local substrate.

Moved verbatim from `src/measure.py` in M2-T06. Every phantom records
`measure_all_baseline`'s output, so the golden checks this move on eight images.

AFM-specific by construction — it needs a Z map in nanometres. The mask geometry
that works on any modality is next door in `geometry.py`; before this task both
lived in one AFM-named module, which is why SEM/TEM code had to import from
`measure` to get shape metrics.

`create_circular_mask` is modality-neutral in principle and stays here in practice:
it has exactly one caller, `measure_all_baseline`, and a module holding one disk
helper would be filing, not structure.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from skimage.filters import threshold_otsu

logger = logging.getLogger(__name__)


def create_circular_mask(shape: tuple, cy: int, cx: int, radius: float) -> np.ndarray:
    """
    Boolean mask: a disk of the given radius centred on (cy, cx).

    Args:
        shape:  (height, width) of the image
        cy, cx: centre, in pixels
        radius: radius, in pixels
    Returns:
        a boolean mask of that shape
    """
    y, x = np.ogrid[: shape[0], : shape[1]]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2


def get_clean_ring(
    mask_particle: np.ndarray, substrate_mask: np.ndarray, outer_px: int, inner_erode_px: int
) -> np.ndarray:
    """
    A ring around a particle, cleared of neighbouring particles.

    The geometric ring is dilation(mask) - erosion(mask); clearing it means
    keeping only substrate pixels (substrate_mask). That removes neighbours
    whether or not they were detected.

    Args:
        mask_particle:   boolean mask of this particle
        substrate_mask:  True where the substrate is (z_above < otsu_threshold)
        outer_px:        outward ring width, in pixels
        inner_erode_px:  inward margin from the mask edge, to clear the slope
    Returns:
        a boolean mask of the cleaned ring
    """
    # Grow the mask by inner_erode_px to get off the particle slope
    mask_expanded = binary_dilation(mask_particle, iterations=inner_erode_px)

    # Build the ring from the grown mask
    outer = binary_dilation(mask_expanded, iterations=outer_px)
    ring = outer & ~mask_expanded

    # Remove neighbours
    clean_ring = ring & substrate_mask

    return clean_ring


def measure_height(
    z_flat: np.ndarray,
    mask_particle: np.ndarray,
    substrate_mask: np.ndarray,
    global_baseline: float,
    outer_px: int = 5,
    inner_erode_px: int = 3,
    min_ring_px: int = 5,
) -> dict:
    """
    Measure the height of a single particle.

    height = max(Z inside the mask) - median(Z in the cleaned ring)

    The baseline comes from z_flat, so the result is in physical nanometres.
    If the ring is too small — dense packing — the global baseline is used
    instead (the median of the whole substrate).

    Args:
        z_flat:          levelled Z map, in nm
        mask_particle:   boolean mask of the particle
        substrate_mask:  True where the substrate is (z_above < otsu_threshold)
        global_baseline: median substrate of the whole image (fallback)
        outer_px:        outward ring width
        inner_erode_px:  inward margin from the mask edge
        min_ring_px:     minimum ring pixels for a trustworthy baseline

    Returns:
        dict: height_nm, baseline_nm, area_px, ring_px, baseline_source
    """
    clean_ring = get_clean_ring(mask_particle, substrate_mask, outer_px, inner_erode_px)

    # Choose the baseline
    if clean_ring.sum() >= min_ring_px:
        baseline = float(np.median(z_flat[clean_ring]))
        baseline_source = "ring"
    else:
        baseline = global_baseline
        baseline_source = "global"

    z_in_mask = z_flat[mask_particle]
    height = float(z_in_mask.max() - baseline)

    return {
        "height_nm": height,
        "mean_nm": float(z_in_mask.mean() - baseline),
        "baseline_nm": baseline,
        "area_px": int(mask_particle.sum()),
        "ring_px": int(clean_ring.sum()),
        "baseline_source": baseline_source,
    }


def measure_all_baseline(
    z_flat: np.ndarray,
    z_above: np.ndarray,
    blobs: np.ndarray,
    outer_px: int = 5,
    inner_erode_px: int = 3,
    min_ring_px: int = 5,
) -> pd.DataFrame:
    """
    Measure every particle's height with the baseline method (circular masks).

    The circular mask is built from the LoG sigma:
        radius_px = sigma * sqrt(2)

    Args:
        z_flat:          levelled Z map, in nm
        z_above:         z_flat - substrate
        blobs:           detect_particles output, (N, 4) [y, x, sigma, radius_nm]
        outer_px:        ring width
        inner_erode_px:  margin from the mask edge
        min_ring_px:     minimum ring pixels

    Returns:
        DataFrame with one row per particle
    """
    # Substrate mask — computed once for the whole image. It accounts for ALL
    # particles, including the ones detection missed.
    otsu_thresh = threshold_otsu(z_above)
    substrate_mask = z_above < otsu_thresh
    global_baseline = float(np.median(z_flat[substrate_mask]))

    logger.debug(
        "global baseline %.3f, Otsu threshold %.3f, substrate %d/%d px",
        global_baseline,
        otsu_thresh,
        substrate_mask.sum(),
        substrate_mask.size,
    )

    results = []

    for i, blob in enumerate(blobs):
        y, x, sigma, radius_nm = blob
        y_i = int(round(y))
        x_i = int(round(x))
        radius_px = sigma * np.sqrt(2)

        # Circular mask, radius from LoG
        mask = create_circular_mask(z_flat.shape, y_i, x_i, radius_px)

        # Edge particles — the mask runs past the image bounds
        if mask.sum() < 4:
            continue

        metrics = measure_height(
            z_flat, mask, substrate_mask, global_baseline, outer_px, inner_erode_px, min_ring_px
        )

        # Discard negative heights — they are artefacts
        if metrics["height_nm"] <= 0:
            continue

        results.append(
            {
                "particle_id": i,
                "x_px": x_i,
                "y_px": y_i,
                "sigma_px": float(sigma),
                "radius_nm": float(radius_nm),
                "method": "baseline_circle",
                **metrics,
            }
        )

    df = pd.DataFrame(results)

    return df
