"""
Детекция частиц на AFM изображениях методом Laplacian of Gaussian (LoG).

- estimate_log_params:     вычисление диапазона sigma из радиусов Otsu
- estimate_log_threshold:  автоматический порог из шума подложки
- detect_particles:        детекция частиц методом LoG
- plot_detections:         визуализация результатов детекции
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.filters import threshold_otsu


def estimate_log_params(sizes: dict) -> dict:
    """
    Вычисляет диапазон sigma для LoG из результатов estimate_radius_otsu.

    Связь радиуса и sigma: radius_px = sigma * sqrt(2)
    Берём диапазон с запасом — LoG сам найдёт оптимальную sigma
    для каждой частицы внутри диапазона.

    Args:
        sizes: dict от estimate_radius_otsu (содержит radii_px)

    Returns:
        dict с min_sigma и max_sigma в пикселях
    """
    radii_px = sizes["radii_px"]

    min_sigma = radii_px.min() / np.sqrt(2) * 0.5   # вдвое меньше минимума
    max_sigma = radii_px.max() / np.sqrt(2) * 2.0   # вдвое больше максимума

    # Защита от вырожденных случаев
    min_sigma = max(min_sigma, 1.0)
    max_sigma = max(max_sigma, min_sigma * 2)

    return {
        "min_sigma": min_sigma,
        "max_sigma": max_sigma,
    }


def estimate_log_threshold(z_above: np.ndarray) -> float:
    """
    Автоматический порог для LoG из шума подложки.

    Пиксели ниже порога Otsu считаем подложкой.
    Порог = 3 * std шума подложки, нормированный на максимум z_above.

    Args:
        z_above: z_flat - substrate (частицы над подложкой)

    Returns:
        порог для blob_log (безразмерный, 0..1)
    """
    otsu_thresh  = threshold_otsu(z_above)
    substrate_px = z_above[z_above < otsu_thresh]
    noise_std    = float(substrate_px.std())
    z_max        = float(z_above.max())

    threshold = 3.0 * noise_std / z_max if z_max > 0 else 0.05

    return threshold

def estimate_log_threshold_adaptive(z_above: np.ndarray,
                                     params: dict,
                                     percentile: float = 20.0) -> float:
    """
    Адаптивный порог из распределения откликов LoG.

    Запускаем LoG с минимальным порогом, смотрим на распределение
    максимальных откликов всех найденных блобов.
    Порог = percentile от этого распределения.

    Устойчив между изображениями — опирается на относительное
    распределение откликов, а не абсолютное значение.

    Args:
        z_above:    z_flat - substrate
        params:     dict от estimate_log_params (min_sigma, max_sigma)
        percentile: нижний процентиль откликов для отсечения шума
                    20 = отсекаем нижние 20% слабых откликов

    Returns:
        адаптивный threshold для blob_log
    """
    z_norm = z_above / z_above.max()

    # Находим все блобы с минимальным порогом
    raw = blob_log(
        z_norm,
        min_sigma=params['min_sigma'],
        max_sigma=params['max_sigma'],
        num_sigma=15,
        threshold=0.01,
        overlap=0.9,
    )

    if len(raw) == 0:
        return 0.05

    # Для каждого блоба — максимальный отклик в окрестности центра
    responses = []
    for blob in raw:
        y, x, sigma = blob
        r  = max(int(sigma), 1)
        y1 = max(0, int(y) - r)
        y2 = min(z_norm.shape[0], int(y) + r)
        x1 = max(0, int(x) - r)
        x2 = min(z_norm.shape[1], int(x) + r)
        responses.append(float(z_norm[y1:y2, x1:x2].max()))

    responses = np.array(responses)
    threshold = float(np.percentile(responses, percentile))

    print(f"   Откликов (min threshold): {len(responses)}")
    print(f"   Отклики:                  "
          f"{responses.min():.3f} – {responses.max():.3f}")
    print(f"   Адаптивный порог ({percentile:.0f}%):  {threshold:.4f}")

    return threshold

def detect_particles(z_above: np.ndarray,
                     pixel_size_nm: float,
                     sizes: dict,
                     overlap: float = 0.3,
                     threshold: float = None,
                     percentile: float = 20.0) -> np.ndarray:
    """
    Детекция частиц методом Laplacian of Gaussian (LoG).

    Алгоритм:
        1. Вычисляем диапазон sigma из радиусов Otsu
        2. Вычисляем порог из шума подложки
        3. Нормируем z_above в [0, 1] — LoG чувствителен к масштабу
        4. Запускаем blob_log
        5. Добавляем физический радиус в нм

    Args:
        z_above:       z_flat - substrate (частицы над подложкой)
        pixel_size_nm: нм/пиксель
        sizes:         dict от estimate_radius_otsu
        overlap:       допустимое перекрытие блобов (0..1)

    Returns:
        blobs: np.ndarray shape (N, 4) — [y, x, sigma_px, radius_nm]
    """
    params    = estimate_log_params(sizes)
    if threshold is None:
        threshold = estimate_log_threshold_adaptive(z_above, params, percentile)

    # LoG работает на нормированном изображении [0, 1]
    z_norm = z_above / z_above.max()

    raw_blobs = blob_log(
        z_norm,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        num_sigma=15,
        threshold=threshold,
        overlap=overlap,
    )

    if len(raw_blobs) == 0:
        print("Частицы не найдены. Попробуй уменьшить threshold.")
        return np.empty((0, 4))

    # radius = sigma * sqrt(2) — стандартная связь для LoG
    sigma_px  = raw_blobs[:, 2]
    radius_nm = sigma_px * np.sqrt(2) * pixel_size_nm

    blobs = np.column_stack([
        raw_blobs[:, :2],   # y, x
        sigma_px,            # sigma в пикселях
        radius_nm,           # радиус в нм
    ])

    blobs = _filter_boundary_blobs(blobs, z_above.shape)

    print(f"✅ Найдено частиц:  {len(blobs)}")
    print(f"   Радиусы:         {radius_nm.min():.1f} – {radius_nm.max():.1f} нм")
    print(f"   Медиана радиуса: {np.median(radius_nm):.1f} нм")
    print(f"   LoG threshold:   {threshold:.4f}")
    print(f"   sigma диапазон:  {params['min_sigma']:.1f} – "
          f"{params['max_sigma']:.1f} пкс")

    return blobs

def _filter_boundary_blobs(blobs: np.ndarray,
                             shape: tuple,
                             margin: float = 1.0) -> np.ndarray:
    """
    Удаляет частицы чей круг выходит за края изображения.

    Args:
        blobs:  (N, 4) — [y, x, sigma, radius_nm]
        shape:  (height, width) изображения
        margin: дополнительный отступ в пикселях (по умолчанию 1)
    Returns:
        отфильтрованный массив blobs
    """
    h, w = shape
    y, x, sigma = blobs[:, 0], blobs[:, 1], blobs[:, 2]
    radius_px = sigma * np.sqrt(2)

    valid = (
        (y - radius_px >= margin)      &   # верхний край
        (y + radius_px <= h - margin)  &   # нижний край
        (x - radius_px >= margin)      &   # левый край
        (x + radius_px <= w - margin)      # правый край
    )

    return blobs[valid]