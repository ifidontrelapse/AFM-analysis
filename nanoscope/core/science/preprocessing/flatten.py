"""Levelling: remove the instrument's tilt and per-line drift from a height map.

Moved from `src/preprocess.py` in M2-T03, verbatim at the time — the algorithms,
constants and order of operations were byte-identical, and the characterization
golden is what proved it. Two things have changed since, each in its own commit:
M2-T12 translated the docstrings, and **M3-T08 (ADR-0029)** made `flatten_lines`
allocate its result at the width it computes in.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq

from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.validation import ensure_height_map


def flatten_plane(z: np.ndarray) -> np.ndarray:
    """
    Remove the overall sample tilt with a least-squares plane fit.

    Args:
        z: 2D array representing the AFM Z-map.
    Returns:
        Flattened Z-map with the best-fit plane removed.
    Raises:
        InvalidImageError: if `z` is not a 2-D, non-empty, numeric, finite array
            (ADR-0030). `scipy.lstsq` rejected the non-finite half of that
            before, in its own words; the rest reached `h, w = z.shape`.
    """
    z = ensure_height_map(z)
    h, w = z.shape
    # Coordinate grids for X and Y
    xi, yi = np.meshgrid(np.arange(w), np.arange(h))
    # Design matrix for the least-squares fit: [X, Y, 1]
    a = np.c_[xi.ravel(), yi.ravel(), np.ones(xi.size)]
    coeffs, *_ = lstsq(a, z.ravel())
    plane = (coeffs[0] * xi + coeffs[1] * yi + coeffs[2]).reshape(h, w)
    return z - plane


def flatten_lines(z: np.ndarray, poly_order: int = 1) -> np.ndarray:
    """
    Per-line levelling: fit and subtract a polynomial trend from every row.

    Args:
        z: sample topography
        poly_order: polynomial degree (default 1 — a linear trend)
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
    """
    z = ensure_height_map(z)
    if poly_order < 0:
        raise InvalidParameterError(f"poly_order must be zero or greater, got {poly_order!r}.")
    if z.shape[1] <= poly_order:
        raise InvalidParameterError(
            f"poly_order={poly_order} needs at least {poly_order + 1} points per row; "
            f"z has {z.shape[1]} column(s)."
        )
    result = np.empty_like(z, dtype=np.promote_types(z.dtype, np.float64))
    xi = np.arange(z.shape[1])
    for i, row in enumerate(z):
        coeffs = np.polyfit(xi, row, poly_order)
        result[i] = row - np.polyval(coeffs, xi)
    return result
