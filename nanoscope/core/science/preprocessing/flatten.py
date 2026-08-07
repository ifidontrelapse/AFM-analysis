"""Levelling: remove the instrument's tilt and per-line drift from a height map.

Moved from `src/preprocess.py` in M2-T03, verbatim at the time — the algorithms,
constants and order of operations were byte-identical, and the characterization
golden is what proved it. Two things have changed since, each in its own commit:
M2-T12 translated the docstrings, and **M3-T08 (ADR-0029)** made `flatten_lines`
allocate its result at the width it computes in.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import lstsq

from nanoscope.core.errors import InvalidImageError, InvalidParameterError
from nanoscope.core.validation import ensure_height_map

logger = logging.getLogger(__name__)


def flatten_plane(z: np.ndarray, *, allow_gaps: bool = False) -> np.ndarray:
    """
    Remove the overall sample tilt with a least-squares plane fit.

    Args:
        z: 2D array representing the AFM Z-map.
        allow_gaps: fit over the finite pixels only, leaving non-finite ones
            absent in the result (ADR-0036). A dropped scan line is a real
            artefact, not malformed input — but it is opt-in, because the
            default contract is ADR-0030's and every other entry point holds
            callers to it.
    Returns:
        Flattened Z-map with the best-fit plane removed. With `allow_gaps`, the
        non-finite pixels of `z` are non-finite here too, in the same places:
        the gap stays **absent** rather than being interpolated into a
        measurement nobody made.
    Raises:
        InvalidImageError: if `z` is not a 2-D, non-empty, numeric, finite array
            (ADR-0030) — the last requirement waived by `allow_gaps`.
            `scipy.lstsq` rejected the non-finite half of that before, in its own
            words; the rest reached `h, w = z.shape`. Also if fewer than three
            pixels are finite, which is not enough to define a plane.
    """
    z = ensure_height_map(z, allow_gaps=allow_gaps)
    h, w = z.shape
    # Coordinate grids for X and Y
    xi, yi = np.meshgrid(np.arange(w), np.arange(h))

    if allow_gaps:
        finite = np.isfinite(z)
        if finite.sum() < 3:
            raise InvalidImageError(
                f"z has {int(finite.sum())} finite pixel(s); a plane needs at least three."
            )
        # The masked fit, and the reason it is not `nan_to_num`: filling the gap
        # with zeros does not add noise, it tells the fit that the sample dips
        # to zero along those lines, and it biases the tilt itself (ADR-0036).
        design = np.c_[xi[finite].ravel(), yi[finite].ravel(), np.ones(int(finite.sum()))]
        coeffs, *_ = lstsq(design, z[finite].ravel())
    else:
        # Design matrix for the least-squares fit: [X, Y, 1]
        a = np.c_[xi.ravel(), yi.ravel(), np.ones(xi.size)]
        coeffs, *_ = lstsq(a, z.ravel())

    plane = (coeffs[0] * xi + coeffs[1] * yi + coeffs[2]).reshape(h, w)
    return z - plane


def flatten_lines(z: np.ndarray, poly_order: int = 1, *, allow_gaps: bool = False) -> np.ndarray:
    """
    Per-line levelling: fit and subtract a polynomial trend from every row.

    Args:
        z: sample topography
        poly_order: polynomial degree (default 1 — a linear trend)
        allow_gaps: fit each row over its finite points only, leaving non-finite
            ones absent in the result (ADR-0036). A row with fewer than
            `poly_order + 1` finite points has no fit and comes back absent in
            full — which is exactly what a dropped scan line is — and the number
            of such rows is warned about, because a scan that lost half its lines
            should not level silently.
    Returns:
        result: levelled topography, in `z`'s dtype promoted with float64 —
            `flatten_plane`'s own promotion rule (D-13, ADR-0029). The residuals
            `polyfit`/`polyval` produce are fractional by construction, so an
            output array narrower than float64 rounds them away — to zeros where
            they were sub-unit, and to *wrapped* values where they were not, a
            negative residual becoming a bright one. A boolean input came back as
            a mask of where the residual was non-zero.
    Raises:
        InvalidImageError: if `z` is not a 2-D, non-empty, numeric, finite array.
        InvalidParameterError: if `poly_order` is negative, or if a row is too
            short to fit it — a polynomial of order k needs k+1 points, and
            `np.polyfit` answered that with `LinAlgError` from inside lstsq.
            With `allow_gaps`, a row that is long enough but too *sparse* is not
            an error: it is a gap, and it comes back absent.
    """
    z = ensure_height_map(z, allow_gaps=allow_gaps)
    if poly_order < 0:
        raise InvalidParameterError(f"poly_order must be zero or greater, got {poly_order!r}.")
    if z.shape[1] <= poly_order:
        raise InvalidParameterError(
            f"poly_order={poly_order} needs at least {poly_order + 1} points per row; "
            f"z has {z.shape[1]} column(s)."
        )
    result = np.empty_like(z, dtype=np.promote_types(z.dtype, np.float64))
    xi = np.arange(z.shape[1])

    if not allow_gaps:
        for i, row in enumerate(z):
            coeffs = np.polyfit(xi, row, poly_order)
            result[i] = row - np.polyval(coeffs, xi)
        return result

    unfitted = 0
    for i, row in enumerate(z):
        finite = np.isfinite(row)
        if finite.sum() <= poly_order:
            # Nothing to fit. The row is absent, not zero: a dropped scan line
            # carries no measurement, and inventing one is the substitution this
            # milestone has spent seven ADRs deleting.
            result[i] = np.nan
            unfitted += 1
            continue
        coeffs = np.polyfit(xi[finite], row[finite], poly_order)
        result[i] = row - np.polyval(coeffs, xi)

    if unfitted:
        logger.warning(
            "%d of %d rows had too few finite points to fit a polynomial of order %d and are "
            "absent in the result; the scan may have lost feedback over that range",
            unfitted,
            z.shape[0],
            poly_order,
        )
    return result
