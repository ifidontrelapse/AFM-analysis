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
    Коррекция общего наклона плоскости методом МНК.

    Args:
        z: 2D array representing the AFM Z-map.
    Returns:
        Flattened Z-map with the best-fit plane removed.
    """
    h, w = z.shape
    # Создаем координатные сетки для X и Y
    xi, yi = np.meshgrid(np.arange(w), np.arange(h))
    # Формируем матрицу A для МНК: [X, Y, 1]
    a = np.c_[xi.ravel(), yi.ravel(), np.ones(xi.size)]
    coeffs, *_ = lstsq(a, z.ravel())
    plane = (coeffs[0] * xi + coeffs[1] * yi + coeffs[2]).reshape(h, w)
    return z - plane


def flatten_lines(z: np.ndarray, poly_order: int = 1) -> np.ndarray:
    """
    Построчное выравнивание, удаление тренда полиномиальной кривой.

    Args:
        z: топология образца
        poly_order: степень полинома для выравнивания (по умолчанию 1 - линейный тренд)
    Returns:
        result: выровненная топология
    """
    result = np.empty_like(z)
    xi = np.arange(z.shape[1])
    for i, row in enumerate(z):
        coeffs = np.polyfit(xi, row, poly_order)
        result[i] = row - np.polyval(coeffs, xi)
    return result
