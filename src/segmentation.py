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
    z_flat: np.ndarray,
    substrate_mask: np.ndarray,
    point_coords: np.ndarray,
    box: np.ndarray,
    outer_ring_px: int,
    inner_erode_px: int,
) -> dict | None:
    """
    Run SAM2 on a single particle and measure its height.
    Returns None if ring is too small (particle is skipped).
    """
    masks_pred, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=np.array([1]),
        box=box,
        multimask_output=True,
    )
    mask = masks_pred[np.argmax(scores)].astype(bool)

    mask_inner = binary_erosion(mask, iterations=inner_erode_px)
    if mask_inner.sum() < 3:
        mask_inner = mask

    ring = binary_dilation(mask, iterations=outer_ring_px) & (~mask) & substrate_mask
    if ring.sum() < 5:
        return None

    baseline = float(np.median(z_flat[ring]))
    peak     = float(z_flat[mask_inner].max())
    cx, cy   = float(point_coords[0, 0]), float(point_coords[0, 1])

    return {
        'x_px':         cx,
        'y_px':         cy,
        'mask':         mask,
        'mask_inner':   mask_inner,
        'ring':         ring,
        'score':        float(np.max(scores)),
        'height_nm':    peak - baseline,
        'baseline_nm':  baseline,
        'peak_nm':      peak,
        'mask_area_px': int(mask.sum()),
    }


def run_sam2_from_blobs(
    predictor,
    z_flat: np.ndarray,
    z_result: np.ndarray,
    blobs: np.ndarray,
    pixel_size_nm: float = 1.0,
    outer_ring_px: int = 5,
    inner_erode_px: int = 2,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    SAM2 segmentation using LoG blob centres as point + box prompts.

    Args:
        predictor:      initialised SAM2ImagePredictor
        z_flat:         flattened Z-map (nm)
        z_result:       z_flat - substrate
        blobs:          (N, 4) array [cy, cx, sigma, r_nm] from detect_particles
        pixel_size_nm:  nm/pixel
        outer_ring_px:  ring width for baseline estimation
        inner_erode_px: mask erosion before peak extraction
    Returns:
        (DataFrame with height measurements, list of mask dicts)
    """
    z_rgb = afm_to_rgb(z_result)
    predictor.set_image(z_rgb)
    substrate_mask = (z_result < threshold_otsu(z_result)).astype(bool)

    records, masks = [], []

    for cy, cx, sigma, r_nm in blobs:
        r   = sigma * np.sqrt(2)
        pad = max(3, r * 0.15)

        res = _run_sam2_single(
            predictor, z_flat, substrate_mask,
            point_coords=np.array([[cx, cy]]),
            box=np.array([cx - r - pad, cy - r - pad,
                          cx + r + pad, cy + r + pad]),
            outer_ring_px=outer_ring_px,
            inner_erode_px=inner_erode_px,
        )
        if res is None:
            continue

        records.append({
            'x_px':          res['x_px'],
            'y_px':          res['y_px'],
            'height_nm':     res['height_nm'],
            'baseline_nm':   res['baseline_nm'],
            'peak_nm':       res['peak_nm'],
            'mask_area_px':  res['mask_area_px'],
            'score':         res['score'],
            'log_radius_nm': r * pixel_size_nm,
        })
        masks.append({k: res[k] for k in
            ('x_px', 'y_px', 'mask', 'mask_inner', 'ring', 'score')})

    return pd.DataFrame(records), masks


def run_sam2_from_boxes(
    predictor,
    z_flat: np.ndarray,
    z_result: np.ndarray,
    boxes_xyxy: np.ndarray,
    outer_ring_px: int = 5,
    inner_erode_px: int = 2,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    SAM2 segmentation using bounding box prompts (from YOLO or any other detector).

    Args:
        predictor:      initialised SAM2ImagePredictor
        z_flat:         flattened Z-map (nm)
        z_result:       z_flat - substrate
        boxes_xyxy:     (N, 4) float — boxes in z_result coordinate space
        outer_ring_px:  ring width for baseline estimation
        inner_erode_px: mask erosion before peak extraction
    Returns:
        (DataFrame with height measurements, list of mask dicts)
    """
    z_rgb = afm_to_rgb(z_result)
    predictor.set_image(z_rgb)
    substrate_mask = (z_result < threshold_otsu(z_result)).astype(bool)

    records, masks = [], []

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        res = _run_sam2_single(
            predictor, z_flat, substrate_mask,
            point_coords=np.array([[cx, cy]]),
            box=np.array([x1, y1, x2, y2]),
            outer_ring_px=outer_ring_px,
            inner_erode_px=inner_erode_px,
        )
        if res is None:
            continue

        records.append({
            'x_px':         res['x_px'],
            'y_px':         res['y_px'],
            'height_nm':    res['height_nm'],
            'baseline_nm':  res['baseline_nm'],
            'peak_nm':      res['peak_nm'],
            'mask_area_px': res['mask_area_px'],
            'sam_score':    res['score'],
        })
        masks.append({k: res[k] for k in
            ('x_px', 'y_px', 'mask', 'mask_inner', 'ring', 'score')}
            | {'box': box})

    return pd.DataFrame(records), masks
