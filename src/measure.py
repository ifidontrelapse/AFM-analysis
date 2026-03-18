"""
Measurement helpers for particle heights and baselines.
"""

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_dilation, binary_erosion


def create_circular_mask(shape: tuple,
                          cy: int, cx: int,
                          radius: float) -> np.ndarray:
    """
    Булева маска - диск радиуса radius с центром (cy, cx).

    Args:
        shape:  (height, width) изображения
        cy, cx: центр в пикселях
        radius: радиус в пикселях
    Returns:
        булева маска shape
    """
    y, x = np.ogrid[:shape[0], :shape[1]]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2


def get_clean_ring(mask_particle: np.ndarray,
                   substrate_mask: np.ndarray,
                   outer_px: int,
                   inner_erode_px: int) -> np.ndarray:
    """
    Кольцо вокруг частицы очищенное от соседних частиц.

    Геометрическое кольцо = dilation(mask) - erosion(mask).
    Очистка = оставляем только пиксели подложки (substrate_mask).
    Это автоматически убирает и задетектированных и незадетектированных соседей.

    Args:
        mask_particle:   булева маска текущей частицы
        substrate_mask:  True там где подложка (z_above < otsu_threshold)
        outer_px:        ширина кольца наружу в пикселях
        inner_erode_px:  отступ от края маски внутрь (уходим со склона)
    Returns:
        булева маска чистого кольца
    """
    # Расширяем маску на inner_erode_px — уходим со склона
    mask_expanded = binary_dilation(mask_particle, iterations=inner_erode_px)

    # Кольцо строим от расширенной маски
    outer = binary_dilation(mask_expanded, iterations=outer_px)
    ring  = outer & ~mask_expanded

    # Убираем соседей
    clean_ring = ring & substrate_mask

    return clean_ring

def measure_height(z_flat: np.ndarray,
                   mask_particle: np.ndarray,
                   substrate_mask: np.ndarray,
                   global_baseline: float,
                   outer_px: int = 5,
                   inner_erode_px: int = 3,
                   min_ring_px: int = 5) -> dict:
    """
    Измерение высоты одной частицы.

    height = max(Z внутри маски) - median(Z в чистом кольце)

    Baseline берём из z_flat — физически правильные нанометры.
    Если кольцо слишком маленькое (плотная упаковка) — используем
    глобальный baseline (медиана всей подложки).

    Args:
        z_flat:          выровненная Z-карта в нм
        mask_particle:   булева маска частицы
        substrate_mask:  True = подложка (z_above < otsu_threshold)
        global_baseline: медиана подложки всего изображения (fallback)
        outer_px:        ширина кольца наружу
        inner_erode_px:  отступ от края маски внутрь
        min_ring_px:     минимум пикселей в кольце для надёжного baseline

    Returns:
        dict: height_nm, baseline_nm, area_px, ring_px, baseline_source
    """
    clean_ring = get_clean_ring(
        mask_particle, substrate_mask, outer_px, inner_erode_px
    )

    # Выбираем baseline
    if clean_ring.sum() >= min_ring_px:
        baseline        = float(np.median(z_flat[clean_ring]))
        baseline_source = "ring"
    else:
        baseline        = global_baseline
        baseline_source = "global"

    z_in_mask = z_flat[mask_particle]
    height    = float(z_in_mask.max() - baseline)

    return {
        "height_nm":       height,
        "mean_nm":         float(z_in_mask.mean() - baseline),
        "baseline_nm":     baseline,
        "area_px":         int(mask_particle.sum()),
        "ring_px":         int(clean_ring.sum()),
        "baseline_source": baseline_source,
    }


def measure_all_baseline(z_flat: np.ndarray,
                          z_above: np.ndarray,
                          blobs: np.ndarray,
                          outer_px: int = 5,
                          inner_erode_px: int = 3,
                          min_ring_px: int = 5) -> pd.DataFrame:
    """
    Измерение высот всех частиц baseline методом (круговые маски).

    Круговая маска строится по sigma из LoG:
        radius_px = sigma * sqrt(2)

    Args:
        z_flat:          выровненная Z-карта в нм
        z_above:         z_flat - substrate
        blobs:           результат detect_particles (N, 4) [y, x, sigma, radius_nm]
        outer_px:        ширина кольца
        inner_erode_px:  отступ от края маски
        min_ring_px:     минимум пикселей в кольце

    Returns:
        DataFrame с результатами для каждой частицы
    """
    # Substrate mask — один раз на всё изображение
    # Содержит ВСЕ частицы включая незадетектированные
    otsu_thresh     = threshold_otsu(z_above)
    substrate_mask  = z_above < otsu_thresh
    global_baseline = float(np.median(z_flat[substrate_mask]))

    print(f"   Global baseline:  {global_baseline:.3f}")
    print(f"   Otsu threshold:   {otsu_thresh:.3f}")
    print(f"   Substrate pixels: {substrate_mask.sum()} / {substrate_mask.size}")

    results = []

    for i, blob in enumerate(blobs):
        y, x, sigma, radius_nm = blob
        y_i = int(round(y))
        x_i = int(round(x))
        radius_px = sigma * np.sqrt(2)

        # Круговая маска по радиусу из LoG
        mask = create_circular_mask(z_flat.shape, y_i, x_i, radius_px)

        # Граничные частицы — маска выходит за пределы изображения
        if mask.sum() < 4:
            continue

        metrics = measure_height(
            z_flat, mask, substrate_mask, global_baseline,
            outer_px, inner_erode_px, min_ring_px
        )

        # Отбрасываем отрицательные высоты — артефакты
        if metrics["height_nm"] <= 0:
            continue

        results.append({
            "particle_id": i,
            "x_px":        x_i,
            "y_px":        y_i,
            "sigma_px":    float(sigma),
            "radius_nm":   float(radius_nm),
            "method":      "baseline_circle",
            **metrics,
        })

    df = pd.DataFrame(results)

    return df