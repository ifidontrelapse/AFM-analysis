"""Substrate estimation: what the surface looks like with the particles removed.

Moved verbatim from `src/preprocess.py` in M2-T03. Same algorithms, same
constants, same order; the golden is the proof. Only whitespace changed.

Three known defects travel with this code and are deliberately **not** fixed here,
because each moves a number the golden records:

- `build_substrate_map` leaves `opening_radius` unbound on the manual-radius
  branch — it is only assigned in the `else`. M3 owns it.
- `min_size_pixel=int(min_size_nm / pixel_size_nm)` is 0 for any scan coarser than
  5 nm/px, which disables the noise filter — audit **D-04**, open decision **B2**.
- `estimate_rough_radius` is annotated `-> int` and can return a float; its
  `print` is a library call that M2-T11 replaces with a log sink.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening as morph_opening

logger = logging.getLogger(__name__)


def get_substrate_map(z: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Estimate the substrate surface with the particles removed.

    A disk of radius radius_px sweeps the substrate, recovering the surface
    topography underneath the particles.

    radius_px must be LARGER than the largest particle radius, in pixels.

    Args:
        z: sample topography
        radius_px: radius in pixels for the morphological opening
    Returns:
        substrate topography
    """
    return morph_opening(z, disk(radius_px)).astype(np.float32)


def estimate_radius_otsu(z_above: np.ndarray, pixel_size_nm: float, min_size_pixel: float) -> dict:
    """
    Estimate the typical particle radius by Otsu thresholding.

    Otsu separates z_above into particles and substrate (z = 0). The median
    radius is used because it tolerates aggregates.

    Args:
        z_above:       z - substrate (particles above the substrate)
        pixel_size_nm: nm per pixel = scan_size_nm / z.shape[0]
        min_size_pixel:   minimum particle size in pixels

    Returns:
        dict with the typical radius, the range, and the number of objects found
    """
    thresh = threshold_otsu(z_above)
    binary = z_above > thresh
    labeled = label(binary)  # merge adjacent pixels into objects
    props = regionprops(labeled)  # object properties, including area

    if len(props) == 0:
        raise ValueError("Otsu found no objects. Check the preprocessing and the image quality.")

    radii_px = np.array([p.equivalent_diameter_area / 2 for p in props])

    # Filter noise immediately — anything smaller than min_size_pixel is not a particle
    valid = radii_px >= min_size_pixel
    radii_px = radii_px[valid]
    radii_nm = radii_px * pixel_size_nm
    radii_nm = radii_px * pixel_size_nm

    typical_radius_px = float(np.median(radii_px))
    typical_radius_nm = float(np.median(radii_nm))

    return {
        "typical_radius_px": typical_radius_px,
        "typical_radius_nm": typical_radius_nm,
        "radii_px": radii_px,
        "radii_nm": radii_nm,
        "n_objects": len(props),
        "otsu_threshold": thresh,
    }


def estimate_rough_radius(
    z: np.ndarray, pixel_size_nm: float, min_size_pixel: float, scale: float = 1.7
) -> int:
    """
    Estimate a starting radius from the image itself, with no hard-coded constants.

    Threshold at median + std, take the median object area, and its square root
    is the rough radius.

    Args:
        z:              the source image
        pixel_size_nm:  nm per pixel = scan_size_nm / z.shape[0]
        min_size_pixel: minimum particle size in pixels — the floor for the estimate
        scale:          radius multiplier, so the disk is safely larger than a
                        particle (default 1.7)

    Returns:
        int: rough radius in pixels for the morphological opening
    """
    z_flat = z.flatten()
    thresh = np.median(z_flat) + z_flat.std()
    binary = z > thresh
    labeled = label(binary)
    props = regionprops(labeled)

    # If nothing was found, fall back to 1% of the image width, in pixels
    if len(props) == 0:
        logger.warning(
            "no objects found for radius estimation — the image is probably too "
            "flat or too noisy; falling back to 1% of the image width"
        )
        return max(int(z.shape[1] * 0.01), min_size_pixel)

    # Median area -> equivalent radius
    median_area = np.median([p.area for p in props])
    radius_px = int(np.sqrt(median_area / np.pi))

    # Scale up so the disk is safely larger than a particle
    rough_radius = max(radius_px * scale, min_size_pixel)

    return rough_radius


def build_substrate_map(
    z: np.ndarray, pixel_size_nm: float, min_size_nm: float = 5, manual_radius_px: float = None
) -> tuple:
    """
    Build the substrate map, estimating the opening radius automatically unless
    one is supplied.

    Args:
        z: sample topography
        pixel_size_nm: nm per pixel = scan_size_nm / z.shape[0]
        min_size_nm: minimum particle size in nm — the floor for the estimate
        manual_radius_px: opening radius, skipping the automatic estimate
        min_radius_px: minimum particle radius in pixels

    Returns:
        substrate:      the substrate map (float32)
        z_above:        z_flat - substrate (particles only)
        opening_radius: the radius finally used, in pixels
        sizes:          dict from estimate_radius_otsu
    """
    # Radius supplied by the caller
    if manual_radius_px is not None:
        substrate = get_substrate_map(z, manual_radius_px)
        z_above = z - substrate
        sizes = estimate_radius_otsu(
            z_above, pixel_size_nm, min_size_pixel=int(min_size_nm / pixel_size_nm)
        )
    # Two-stage estimate: rough approximation -> Otsu, floored by the minimum
    # particle size (5 nm by default)
    else:
        # Rough radius approximation
        rough_radius = estimate_rough_radius(
            z, pixel_size_nm, min_size_pixel=int(min_size_nm / pixel_size_nm)
        )
        rough_substrate = get_substrate_map(z, radius_px=rough_radius)
        z_above_rough = z - rough_substrate

        # Refine the radius with Otsu
        sizes = estimate_radius_otsu(
            z_above_rough, pixel_size_nm, min_size_pixel=int(min_size_nm / pixel_size_nm)
        )
        opening_radius = max(int(sizes["typical_radius_px"] * 2.5), 5)

        # Final topography with the substrate subtracted
        substrate = get_substrate_map(z, opening_radius)
        z_above = z - substrate

    return substrate, z_above, opening_radius, sizes
