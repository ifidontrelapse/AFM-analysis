"""
Препроцессинг AFM данных: выравнивание плоскости, удаление тренда, выделение частиц от фона.
- flatten_plane: удаление общего наклона плоскости методом МНК.
- flatten_lines: построчное выравнивание, удаление тренда полиномиальной кривой.
- get_substrate_map: оценка фона (подложки) методом морфологического открытия с 
  большим диском.
- subtract_substrate: выделение карты частиц путем вычитания оценки подложки из исходной
  карты Z.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq
from skimage.morphology import disk, opening as morph_opening


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
        z: 2D array representing the AFM Z-map.
        poly_order: Degree of the polynomial to fit (default is 1 for linear).
    Returns:
        Flattened Z-map with the fitted polynomial trend removed from each line.
    """
    result = np.empty_like(z)
    xi = np.arange(z.shape[1])
    for i, row in enumerate(z):
        coeffs = np.polyfit(xi, row, poly_order)
        result[i] = row - np.polyval(coeffs, xi)
    return result


def get_substrate_map(z: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Estimate substrate via morphological opening using a large disk.
    
    """
    selem = disk(radius_px)
    return morph_opening(z, selem)


def subtract_substrate(z: np.ndarray, radius_px: int) -> np.ndarray:
    """Return particles-only map by subtracting the substrate estimate."""
    substrate = get_substrate_map(z, radius_px)
    return z - substrate
