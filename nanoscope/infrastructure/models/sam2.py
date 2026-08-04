"""SAM2 segmentation runners — an adapter around a model the caller supplies.

Moved verbatim from `src/segmentation.py` in M2-T07, minus the two rendering
helpers, which went to `infrastructure.imaging` where colormaps belong.

The `predictor` arrives as an argument rather than being constructed here, which
is why this module imports no torch at all: the original author isolated it
deliberately ("so the rest of the pipeline can run without SAM installed"), and
that decision is what made M1-T08's torch-free CI environment possible.

One line is not verbatim: `_run_sam2_single` imported `src.measure` inside the
function and now imports `nanoscope.core.science.measurement`. Leaving it would
have pointed an adapter at the legacy shim — infrastructure depending on `src`,
backwards for the layer and a cycle waiting for M2-T09.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.filters import threshold_otsu

# The runners call `afm_to_rgb` to build SAM2's input. It left for
# `infrastructure.imaging` in the same commit — ruff caught the dangling
# reference (F821) before the tests ran, which is the argument for keeping
# `ruff check` blocking on moved code instead of excluding it like `src/`.
from nanoscope.infrastructure.imaging import afm_to_rgb


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
    from nanoscope.core.science.measurement import measure_geometry_from_mask

    masks_pred, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=np.array([1]),
        box=box,
        multimask_output=True,
    )
    mask = masks_pred[np.argmax(scores)].astype(bool)
    cx = float(point_coords[0, 0])
    cy = float(point_coords[0, 1])
    score = float(np.max(scores))

    if z_flat is not None:
        # ── AFM: ring-baseline height ─────────────────────────────────────────
        mask_inner = binary_erosion(mask, iterations=inner_erode_px)
        if mask_inner.sum() < 3:
            mask_inner = mask

        ring = binary_dilation(mask, iterations=outer_ring_px) & (~mask) & substrate_mask
        if ring.sum() < 5:
            return None

        baseline = float(np.median(z_flat[ring]))
        peak = float(z_flat[mask_inner].max())

        return {
            "x_px": cx,
            "y_px": cy,
            "mask": mask,
            "mask_inner": mask_inner,
            "ring": ring,
            "score": score,
            "height_nm": peak - baseline,
            "baseline_nm": baseline,
            "peak_nm": peak,
            "mask_area_px": int(mask.sum()),
        }

    # ── SEM/TEM: geometry from mask ───────────────────────────────────────
    geom = measure_geometry_from_mask(mask, nm_per_pixel)
    if geom["area_px"] == 0:
        return None

    return {
        "x_px": cx,
        "y_px": cy,
        "mask": mask,
        "score": score,
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
    substrate_mask = (image < threshold_otsu(image)).astype(bool) if z_flat is not None else None

    records, masks = [], []

    for cy, cx, sigma, r_nm in blobs:
        r = sigma * np.sqrt(2)
        pad = max(3, r * 0.15)

        res = _run_sam2_single(
            predictor,
            z_flat,
            substrate_mask,
            nm_per_pixel,
            point_coords=np.array([[cx, cy]]),
            box=np.array([cx - r - pad, cy - r - pad, cx + r + pad, cy + r + pad]),
            outer_ring_px=outer_ring_px,
            inner_erode_px=inner_erode_px,
        )
        if res is None:
            continue

        record = {"x_px": res["x_px"], "y_px": res["y_px"], "score": res["score"]}
        for k in (
            "height_nm",
            "baseline_nm",
            "peak_nm",
            "mask_area_px",
            "area_nm2",
            "radius_nm",
            "circularity",
            "aspect_ratio",
            "area_px",
        ):
            if k in res:
                record[k] = res[k]
        if z_flat is not None:
            record["log_radius_nm"] = r * nm_per_pixel if nm_per_pixel else None
        records.append(record)

        mask_entry = {
            "x_px": res["x_px"],
            "y_px": res["y_px"],
            "mask": res["mask"],
            "score": res["score"],
        }
        if "mask_inner" in res:
            mask_entry["mask_inner"] = res["mask_inner"]
            mask_entry["ring"] = res["ring"]
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
    substrate_mask = (image < threshold_otsu(image)).astype(bool) if z_flat is not None else None

    records, masks = [], []

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        res = _run_sam2_single(
            predictor,
            z_flat,
            substrate_mask,
            nm_per_pixel,
            point_coords=np.array([[cx, cy]]),
            box=np.array([x1, y1, x2, y2]),
            outer_ring_px=outer_ring_px,
            inner_erode_px=inner_erode_px,
        )
        if res is None:
            continue

        record = {"x_px": res["x_px"], "y_px": res["y_px"], "sam_score": res["score"]}
        for k in (
            "height_nm",
            "baseline_nm",
            "peak_nm",
            "mask_area_px",
            "area_nm2",
            "radius_nm",
            "circularity",
            "aspect_ratio",
            "area_px",
        ):
            if k in res:
                record[k] = res[k]
        records.append(record)

        mask_entry = {
            "x_px": res["x_px"],
            "y_px": res["y_px"],
            "mask": res["mask"],
            "score": res["score"],
            "box": box,
        }
        if "mask_inner" in res:
            mask_entry["mask_inner"] = res["mask_inner"]
            mask_entry["ring"] = res["ring"]
        masks.append(mask_entry)

    return pd.DataFrame(records), masks
