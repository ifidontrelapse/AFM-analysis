"""Substrate estimation: what the surface looks like with the particles removed.

Moved verbatim from `src/preprocess.py` in M2-T03. Same algorithms, same
constants, same order; the golden is the proof. Only whitespace changed.

Closed here: **D-01** (M3-T01, ADR-0014), **D-05 / D-06** (M3-T06, ADR-0017),
**D-10** (M3-T09, ADR-0020) — every radius reaching `disk()` is an integer,
rounded up, and `estimate_rough_radius` no longer lies about its return type —
and **D-04** (M3-T02, ADR-0024): the minimum particle size is compared in
nanometres, so `int(min_size_nm / pixel_size_nm)`, which was 0 on 90 % of real
scans and disabled the noise filter, is gone. The duplicated `radii_nm`
assignment (audit §Duplication) went with it: the second line was the one that
had to move above the filter.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening as morph_opening

logger = logging.getLogger(__name__)


def _integer_radius(radius_px: float) -> int:
    """The integer radius `disk()` can centre, rounded **up** (ADR-0020 / D-10).

    `disk(8.5)` is an 18x18 element with no centre pixel, so the opening is
    biased by half a pixel. Any integer gives `2r+1`, which is always odd and
    always centred — so the decision is only which way to round, and rounding up
    is the one that never leaves a particle inside the substrate.
    """
    return int(np.ceil(radius_px))


def get_substrate_map(z: np.ndarray, radius_px: float) -> np.ndarray:
    """
    Estimate the substrate surface with the particles removed.

    A disk of radius radius_px sweeps the substrate, recovering the surface
    topography underneath the particles.

    radius_px must be LARGER than the largest particle radius, in pixels.

    Args:
        z: sample topography
        radius_px: radius in pixels for the morphological opening; rounded up to
            an integer, because `disk()` centres nothing else (ADR-0020)
    Returns:
        substrate topography
    """
    return morph_opening(z, disk(_integer_radius(radius_px))).astype(np.float32)


def estimate_radius_otsu(
    z_above: np.ndarray, pixel_size_nm: float | None, min_size_nm: float
) -> dict:
    """
    Estimate the typical particle radius by Otsu thresholding.

    Otsu separates z_above into particles and substrate (z = 0). The median
    radius is used because it tolerates aggregates.

    Args:
        z_above:       z - substrate (particles above the substrate)
        pixel_size_nm: nm per pixel = scan_size_nm / z.shape[0], or `None` when
            the scale is unknown — then the nanometre outputs are `None` and the
            `min_size_nm` filter cannot be applied (ADR-0025)
        min_size_nm:   minimum particle radius, in nanometres. Compared against
            the radii **in nanometres** (ADR-0024 / D-04); there is no pixel
            conversion and therefore no floor to zero on a coarse scan.

    Returns:
        dict with the typical radius, the range, and `n_objects` — the number of
        objects that **survived** the `min_size_nm` filter, which is the same
        length as `radii_px` (ADR-0017 / D-06). `radii_nm` and
        `typical_radius_nm` are `None` when `pixel_size_nm` is.

    Raises:
        ValueError: if Otsu finds no objects at all, or if the filter removes
            every one of them (ADR-0017 / D-05). The second case used to return
            `nan` and fail several calls later.
    """
    thresh = threshold_otsu(z_above)
    binary = z_above > thresh
    labeled = label(binary)  # merge adjacent pixels into objects
    props = regionprops(labeled)  # object properties, including area

    if len(props) == 0:
        raise ValueError("Otsu found no objects. Check the preprocessing and the image quality.")

    radii_px = np.array([p.equivalent_diameter_area / 2 for p in props])

    if pixel_size_nm is None:
        # No scale, so `min_size_nm` cannot be expressed in anything this image
        # has. Skipping the filter is the honest answer — the pixel-space work
        # below is unaffected — but skipping it *silently* is D-04 again, so it
        # is said out loud, once, here (ADR-0025).
        logger.warning(
            "no physical scale: the %s nm minimum particle size cannot be applied, so every "
            "object Otsu found is kept — on a noisy scan the radius estimate is then driven "
            "by single-pixel noise",
            min_size_nm,
        )
        return {
            "typical_radius_px": float(np.median(radii_px)),
            "typical_radius_nm": None,
            "radii_px": radii_px,
            "radii_nm": None,
            "n_objects": len(radii_px),
            "otsu_threshold": thresh,
        }

    radii_nm = radii_px * pixel_size_nm

    n_found, largest_nm = len(radii_px), float(radii_nm.max())
    # Filter noise immediately, comparing nanometres with nanometres: the old
    # `int(min_size_nm / pixel_size_nm)` was 0 on 90% of real scans and disabled
    # the filter entirely (ADR-0024 / D-04).
    keep = radii_nm >= min_size_nm
    radii_px, radii_nm = radii_px[keep], radii_nm[keep]

    if radii_px.size == 0:
        raise ValueError(
            f"Otsu found {n_found} objects, none with a radius of at least "
            f"min_size_nm={min_size_nm} nm (the largest is {largest_nm:.3g} nm). "
            "Lower the minimum size, or check the preprocessing and the image quality."
        )

    typical_radius_px = float(np.median(radii_px))
    typical_radius_nm = float(np.median(radii_nm))

    return {
        "typical_radius_px": typical_radius_px,
        "typical_radius_nm": typical_radius_nm,
        "radii_px": radii_px,
        "radii_nm": radii_nm,
        "n_objects": len(radii_px),  # post-filter, and so equal to len(radii_px) (D-06)
        "otsu_threshold": thresh,
    }


def estimate_rough_radius(
    z: np.ndarray, pixel_size_nm: float | None, min_size_nm: float, scale: float = 1.7
) -> int:
    """
    Estimate a starting radius from the image itself, with no hard-coded constants.

    Threshold at median + std, take the median object area, and its square root
    is the rough radius.

    Args:
        z:              the source image
        pixel_size_nm:  nm per pixel = scan_size_nm / z.shape[0], or `None` when
                        the scale is unknown — then the floor is 0 px, because a
                        nanometre floor cannot be converted (ADR-0025)
        min_size_nm:    minimum particle radius in nm — the floor for the
                        estimate. Converted to pixels here, and **not** floored
                        to an integer on the way (ADR-0024)
        scale:          radius multiplier, so the disk is safely larger than a
                        particle (default 1.7)

    Returns:
        int: rough radius in pixels for the morphological opening
    """
    # 0.0, not `min_size_nm`: without a scale the floor is not "the same number
    # in pixels" — that is the unit confusion ADR-0024 deleted. It is nothing.
    min_size_px = 0.0 if pixel_size_nm is None else min_size_nm / pixel_size_nm

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
        return _integer_radius(max(z.shape[1] * 0.01, min_size_px))

    # Median area -> equivalent radius
    median_area = np.median([p.area for p in props])
    radius_px = int(np.sqrt(median_area / np.pi))

    # Scale up so the disk is safely larger than a particle
    rough_radius = max(radius_px * scale, min_size_px)

    return _integer_radius(rough_radius)


def build_substrate_map(
    z: np.ndarray,
    pixel_size_nm: float | None,
    min_size_nm: float = 5,
    manual_radius_px: float = None,
) -> tuple:
    """
    Build the substrate map, estimating the opening radius automatically unless
    one is supplied.

    Args:
        z: sample topography
        pixel_size_nm: nm per pixel = scan_size_nm / z.shape[0], or `None` when
            the scale is unknown. Every pixel-space result is unchanged by that;
            the `sizes` dict's `_nm` entries are `None` and the `min_size_nm`
            filter is not applied, with a warning (ADR-0025)
        min_size_nm: minimum particle radius in nm — the floor for the estimate,
            and the noise filter in `estimate_radius_otsu`. Since ADR-0024 it is
            used as a physical size at both sites; nothing converts it with `int()`

    Returns:
        substrate:      the substrate map (float32)
        z_above:        z_flat - substrate (particles only)
        opening_radius: the radius finally used, in pixels
        sizes:          dict from estimate_radius_otsu
    """
    # Radius supplied by the caller
    if manual_radius_px is not None:
        # Rounded up, and *reported* rounded up: ADR-0014 made this branch return
        # the radius it actually uses, and since ADR-0020 the radius it uses is
        # an integer. Returning the caller's 8.5 while opening with 9 would put
        # the lie back, one field further along.
        opening_radius = _integer_radius(manual_radius_px)
        substrate = get_substrate_map(z, opening_radius)
        z_above = z - substrate
        sizes = estimate_radius_otsu(z_above, pixel_size_nm, min_size_nm=min_size_nm)
    # Two-stage estimate: rough approximation -> Otsu, floored by the minimum
    # particle radius (5 nm by default)
    else:
        # Rough radius approximation
        rough_radius = estimate_rough_radius(z, pixel_size_nm, min_size_nm=min_size_nm)
        rough_substrate = get_substrate_map(z, radius_px=rough_radius)
        z_above_rough = z - rough_substrate

        # Refine the radius with Otsu
        sizes = estimate_radius_otsu(z_above_rough, pixel_size_nm, min_size_nm=min_size_nm)
        opening_radius = max(_integer_radius(sizes["typical_radius_px"] * 2.5), 5)

        # Final topography with the substrate subtracted
        substrate = get_substrate_map(z, opening_radius)
        z_above = z - substrate

    return substrate, z_above, opening_radius, sizes
