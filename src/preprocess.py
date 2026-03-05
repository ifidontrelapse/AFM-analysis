"""
Препроцессинг AFM данных: выравнивание плоскости, удаление тренда, выделение частиц от фона.
- flatten_plane: удаление общего наклона плоскости методом МНК.
- flatten_lines: построчное выравнивание, удаление тренда полиномиальной кривой.
- get_substrate_map: оценка поверхности подложки без частиц через morphological opening.
- estimate_radius_otsu: оценка типичного радиуса частиц через бинаризацию Otsu и анализ объектов.
- estimate_rough_radius: грубая оценка радиуса крупных частиц из изображения.
- build_substrate_map: построение карты подложки с возможностью автоматической оценки радиуса для morphological opening.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
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


def get_substrate_map(z: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Оценка поверхности подложки без частиц.
    
    Диск радиуса radius_px заметает подложку,
    получая точную топологию поверхности.
    
    radius_px должен быть БОЛЬШЕ радиуса самой крупной частицы в пикселях.

    Args:
        z: топология образца
        radius_px: радиус в пикселях для morphological opening
    Returns:
        топология подложки
    """
    return morph_opening(z, disk(radius_px)).astype(np.float32)


def estimate_radius_otsu(z_above: np.ndarray,
                          pixel_size_nm: float) -> dict:
    """
    Оценка типичного радиуса частиц через бинаризацию Otsu.
    
    Otsu разделяет z_above на частицы и подложку (z = 0).
    Медиана радиусов должна быть устойчива к агрегатам.
    
    Args:
        z_above:       z - substrate (частицы над подложкой)
        pixel_size_nm: нм/пиксель = scan_size_nm / z.shape[0]
    
    Returns:
        dict с типичным радиусом, диапазоном и числом найденных объектов
    """
    thresh  = threshold_otsu(z_above)
    binary  = z_above > thresh
    labeled = label(binary)        # Объединение соседних пикселей в объекты
    props   = regionprops(labeled) # Получение свойств объектов, включая площадь

    if len(props) == 0:
        raise ValueError(
            "Otsu не нашёл ни одного объекта."
            "Проверьте качество предобработки и изображения."
        )

    radii_px = np.array([p.equivalent_diameter_area / 2 for p in props])
    radii_nm = radii_px * pixel_size_nm

    typical_radius_px = float(np.median(radii_px))
    typical_radius_nm = float(np.median(radii_nm))

    return {
        "typical_radius_px":  typical_radius_px,
        "typical_radius_nm":  typical_radius_nm,
        "radii_px":           radii_px,
        "radii_nm":           radii_nm,
        "n_objects":          len(props),
        "otsu_threshold":     thresh,
    }

def estimate_rough_radius(z: np.ndarray, pixel_size_nm: float, min_size_pixel: float, scale: float = 1.7) -> int:
    """
    Оценка стартового радиуса из изображения без констант.
    
    Берём простой порог (медиана + std),
    считаем среднюю площадь объектов и берём корень - грубый радиус.

    Args:
        z:              исходное изображение
        pixel_size_nm:  нм/пиксель = scan_size_nm / z.shape[0]
        min_size_pixel: минимальный размер частицы в пикселях (для ограничения радиуса при автоматической оценке)
        scale:          множитель для радиуса, чтобы диск был заведомо больше частицы (по умолчанию 1.7)
    
    Returns:
        int: грубый радиус в пикселях для morphological opening
    """
    z_flat = z.flatten()
    thresh  = np.median(z_flat) + z_flat.std()
    binary  = z > thresh
    labeled = label(binary)
    props   = regionprops(labeled)

    # В случае, если ничего не найдено, возвращаем 1% от размера изображения в пикселях
    if len(props) == 0:
        print("Warning: Не найдено объектов для оценки радиуса. Вероятно, изображение слишком ровное или зашумленное." \
        "\nПо умолчанию использован радиус, равный 1% от размера изображения.")
        return max(int(z_flat.shape[1] * 0.01), min_size_pixel)

    # Медиана площадей -> эквивалентный радиус
    median_area   = np.median([p.area for p in props])
    radius_px     = int(np.sqrt(median_area / np.pi))

    # Умножаем на scale, чтобы диск был заведомо больше частицы
    rough_radius  = max(radius_px * scale, min_size_pixel)

    return rough_radius


def build_substrate_map(z: np.ndarray,
                         pixel_size_nm: float,
                         min_size_nm: float = 5,
                         manual_radius_px: float = None) -> tuple:
    """
    Построение карты подложки с возможностью автоматической оценки радиуса для morphological opening.
    
    Args:
        z: топология образца        
        pixel_size_nm: нм/пиксель = scan_size_nm / z.shape[0]
        min_size_nm: минимальный размер частицы в нм (для ограничения радиуса при автоматической оценке)
        manual_radius_px: радиус для opening без автоматической оценки
    
    Returns:
        substrate:      карта подложки (float32)
        z_above:        z_flat - substrate (только частицы)
        opening_radius: итоговый радиус в пикселях
        sizes:          dict от estimate_radius_otsu
    """
    # Ручное задание радиуса
    if manual_radius_px is not None:
        substrate = get_substrate_map(z, manual_radius_px)
        z_above   = z - substrate
        sizes = estimate_radius_otsu(z_above, pixel_size_nm)
    # Двухстадийная оценка радиуса через грубое приближение -> Otsu, исходя из минимального радиуса частиц (по умолчанию 5 нм)
    else:
        # Грубое приближение радиуса
        rough_radius  = estimate_rough_radius(z, pixel_size_nm, min_size_pixel=int(min_size_nm / pixel_size_nm))
        rough_substrate = get_substrate_map(z, radius_px=rough_radius)
        z_above_rough   = z - rough_substrate

        # Оценка радиуса через Otsu
        sizes = estimate_radius_otsu(z_above_rough, pixel_size_nm)
        opening_radius = max(int(sizes["typical_radius_px"] * 2.5), 5)

        # Финальная топология с вычетом подложки
        substrate = get_substrate_map(z, opening_radius)
        z_above   = z - substrate

    return substrate, z_above, opening_radius, sizes

