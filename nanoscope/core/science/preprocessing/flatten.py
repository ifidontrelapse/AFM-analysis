"""Levelling: remove the instrument's tilt and per-line drift from a height map.

Moved verbatim from `src/preprocess.py` in M2-T03 — the algorithms, constants and
order of operations are byte-identical, and the characterization golden is what
proves it. Only whitespace changed, by `ruff format`.

The Russian docstrings come across untranslated on purpose: M2-T12 owns that, and
mixing a translation into a move would make a red golden ambiguous.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq


def flatten_plane(z: np.ndarray) -> np.ndarray:
    """
    Remove the overall sample tilt with a least-squares plane fit.

    Args:
        z: 2D array representing the AFM Z-map.
    Returns:
        Flattened Z-map with the best-fit plane removed.
    """
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
        result: levelled topography
    """
    result = np.empty_like(z)
    xi = np.arange(z.shape[1])
    for i, row in enumerate(z):
        coeffs = np.polyfit(xi, row, poly_order)
        result[i] = row - np.polyval(coeffs, xi)
    return result
