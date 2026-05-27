"""AFM segmentation helpers (SAM2-based).

These functions are intentionally isolated so the rest of the pipeline can
run without SAM installed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_dilation, binary_erosion


def afm_to_rgb(z: np.ndarray, colormap: str = "afmhot", clip_percentile: float = 99.0) -> np.ndarray:
    """Convert a Z-map (float) into uint8 RGB for SAM input."""
    lo, hi = z.min(), np.percentile(z, clip_percentile)
    z_clip = np.clip(z, lo, hi)
    z_norm = (z_clip - lo) / (hi - lo + 1e-9)
    cmap = plt.get_cmap(colormap)
    rgb = (cmap(z_norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def overlay_masks(rgb_img: np.ndarray, sam_results: list[dict], alpha: float = 0.45) -> np.ndarray:
    """Overlay SAM masks with random colors for visualization."""
    overlay = rgb_img.copy().astype(float)
    rng = np.random.default_rng(0)
    for r in sam_results:
        color = rng.integers(80, 255, 3).astype(float)
        for c in range(3):
            overlay[:, :, c][r["mask"]] = (
                alpha * color[c] + (1 - alpha) * overlay[:, :, c][r["mask"]]
            )
    return overlay.clip(0, 255).astype(np.uint8)


def _run_sam2_single(
    predictor,
    z_flat: np.ndarray | None,
    substrate_mask: np.ndarray | None,
    nm_per_pixel: float | None,
    point_coords: np.ndarray,
    box: np.ndarray,
    outer_ring_px: int,
    inner_erode_px: int,
) -> dict | None:
    """
    Run SAM2 on a single particle.

    AFM  (z_flat is not None): ring-baseline height measurement.
    SEM/TEM (z_flat is None):  geometry from mask.

    Returns None if the particle should be skipped.
    """
    from src.measure import measure_geometry_from_mask

    masks_pred, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=np.array([1]),
        box=box,
        multimask_output=True,
    )
    mask  = masks_pred[np.argmax(scores)].astype(bool)
    cx    = float(point_coords[0, 0])
    cy    = float(point_coords[0, 1])
    score = float(np.max(scores))

    if z_flat is not None:
        # ── AFM: ring-baseline height ─────────────────────────────────────────
        mask_inner = binary_erosion(mask, iterations=inner_erode_px)
        if mask_inner.sum() < 3:
            mask_inner = mask

        ring = (
            binary_dilation(mask, iterations=outer_ring_px)
            & (~mask)
            & substrate_mask
        )
        if ring.sum() < 5:
            return None

        baseline = float(np.median(z_flat[ring]))
        peak     = float(z_flat[mask_inner].max())

        return {
            'x_px': cx, 'y_px': cy,
            'mask': mask, 'mask_inner': mask_inner, 'ring': ring,
            'score': score,
            'height_nm':    peak - baseline,
            'baseline_nm':  baseline,
            'peak_nm':      peak,
            'mask_area_px': int(mask.sum()),
        }

    else:
        # ── SEM/TEM: geometry from mask ───────────────────────────────────────
        geom = measure_geometry_from_mask(mask, nm_per_pixel)
        if geom['area_px'] == 0:
            return None

        return {
            'x_px': cx, 'y_px': cy,
            'mask': mask, 'score': score,
            **geom,
        }


def run_sam2_from_blobs(
    predictor,
    z_flat: np.ndarray | None,
    image: np.ndarray,
    blobs: np.ndarray,
    nm_per_pixel: float | None = None,
    outer_ring_px: int = 5,
    inner_erode_px: int = 2,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    SAM2 segmentation using LoG blob centres as point + box prompts.

    Args:
        predictor:      initialised SAM2ImagePredictor
        z_flat:         flattened Z-map (nm) — None for SEM/TEM
        image:          z_result for AFM; raw image array for SEM/TEM
        blobs:          (N, 4) array [cy, cx, sigma, r_nm] from detect_particles
        nm_per_pixel:   nm/pixel (None if unknown)
        outer_ring_px:  ring width for AFM baseline estimation
        inner_erode_px: mask erosion before AFM peak extraction
    Returns:
        (DataFrame with measurements, list of mask dicts)
    """
    z_rgb = afm_to_rgb(image)
    predictor.set_image(z_rgb)
    substrate_mask = (
        (image < threshold_otsu(image)).astype(bool)
        if z_flat is not None else None
    )

    records, masks = [], []

    for cy, cx, sigma, r_nm in blobs:
        r   = sigma * np.sqrt(2)
        pad = max(3, r * 0.15)

        res = _run_sam2_single(
            predictor, z_flat, substrate_mask, nm_per_pixel,
            point_coords=np.array([[cx, cy]]),
            box=np.array([cx - r - pad, cy - r - pad,
                          cx + r + pad, cy + r + pad]),
            outer_ring_px=outer_ring_px,
            inner_erode_px=inner_erode_px,
        )
        if res is None:
            continue

        record = {'x_px': res['x_px'], 'y_px': res['y_px'], 'score': res['score']}
        for k in ('height_nm', 'baseline_nm', 'peak_nm', 'mask_area_px',
                  'area_nm2', 'radius_nm', 'circularity', 'aspect_ratio', 'area_px'):
            if k in res:
                record[k] = res[k]
        if z_flat is not None:
            record['log_radius_nm'] = r * nm_per_pixel if nm_per_pixel else None
        records.append(record)

        mask_entry = {'x_px': res['x_px'], 'y_px': res['y_px'],
                      'mask': res['mask'], 'score': res['score']}
        if 'mask_inner' in res:
            mask_entry['mask_inner'] = res['mask_inner']
            mask_entry['ring']       = res['ring']
        masks.append(mask_entry)

    return pd.DataFrame(records), masks


def run_sam2_from_boxes(
    predictor,
    z_flat: np.ndarray | None,
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    nm_per_pixel: float | None = None,
    outer_ring_px: int = 5,
    inner_erode_px: int = 2,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    SAM2 segmentation using bounding box prompts (from YOLO or any other detector).

    Args:
        predictor:      initialised SAM2ImagePredictor
        z_flat:         flattened Z-map (nm) — None for SEM/TEM
        image:          z_result for AFM; raw image array for SEM/TEM
        boxes_xyxy:     (N, 4) float — boxes in image coordinate space
        nm_per_pixel:   nm/pixel (None if unknown)
        outer_ring_px:  ring width for AFM baseline estimation
        inner_erode_px: mask erosion before AFM peak extraction
    Returns:
        (DataFrame with measurements, list of mask dicts)
    """
    z_rgb = afm_to_rgb(image)
    predictor.set_image(z_rgb)
    substrate_mask = (
        (image < threshold_otsu(image)).astype(bool)
        if z_flat is not None else None
    )

    records, masks = [], []

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        res = _run_sam2_single(
            predictor, z_flat, substrate_mask, nm_per_pixel,
            point_coords=np.array([[cx, cy]]),
            box=np.array([x1, y1, x2, y2]),
            outer_ring_px=outer_ring_px,
            inner_erode_px=inner_erode_px,
        )
        if res is None:
            continue

        record = {'x_px': res['x_px'], 'y_px': res['y_px'], 'sam_score': res['score']}
        for k in ('height_nm', 'baseline_nm', 'peak_nm', 'mask_area_px',
                  'area_nm2', 'radius_nm', 'circularity', 'aspect_ratio', 'area_px'):
            if k in res:
                record[k] = res[k]
        records.append(record)

        mask_entry = {'x_px': res['x_px'], 'y_px': res['y_px'],
                      'mask': res['mask'], 'score': res['score'],
                      'box': box}
        if 'mask_inner' in res:
            mask_entry['mask_inner'] = res['mask_inner']
            mask_entry['ring']       = res['ring']
        masks.append(mask_entry)

    return pd.DataFrame(records), masks
