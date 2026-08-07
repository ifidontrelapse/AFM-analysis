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
from nanoscope.core.science.measurement.schema import empty_measurement_table, measurement_columns
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
            # `mask_score`, not `score`: since ADR-0028 this project has two
            # scores, and `Detection.confidence` is the detector's. This one is
            # SAM2's predicted IoU for the mask it just produced (ADR-0031).
            "mask_score": score,
            "height_nm": peak - baseline,
            "baseline_nm": baseline,
            "peak_nm": peak,
            "mean_nm": float(z_flat[mask_inner].mean()) - baseline,
            # The ring is the only baseline this path has: a particle whose ring
            # is too small is skipped above rather than falling back to a global
            # median, which is where it differs from `measure_all_baseline`.
            "baseline_source": "ring",
            "ring_px": int(ring.sum()),
            # Was `mask_area_px`. The same count of pixels the baseline producer
            # calls `area_px` (D-17).
            "area_px": int(mask.sum()),
        }

    # ── SEM/TEM: geometry from mask ───────────────────────────────────────
    geom = measure_geometry_from_mask(mask, nm_per_pixel)
    if geom["area_px"] == 0:
        return None

    return {
        "x_px": cx,
        "y_px": cy,
        "mask": mask,
        "mask_score": score,
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

    # A LoG blob prompted this, so the detector block is available; the height
    # block needs a z map and the geometry block needs a real mask to measure.
    blocks = {
        "detector": True,
        "segmentation": True,
        "height": z_flat is not None,
        "geometry": z_flat is None,
    }
    columns = measurement_columns(**blocks)

    records: list[dict] = []
    masks: list[dict] = []

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

        # Built from the declared block, not with `if k in res`: that was how
        # two particles in one call ended up with different columns (D-17).
        res["particle_id"] = len(records)
        res["method"] = "sam2_blobs"
        res["sigma_px"] = float(sigma)
        res["detector_radius_nm"] = r * nm_per_pixel if nm_per_pixel else None
        records.append({name: res[name] for name in columns})

        mask_entry = {
            "x_px": res["x_px"],
            "y_px": res["y_px"],
            "mask": res["mask"],
            "mask_score": res["mask_score"],
        }
        if "mask_inner" in res:
            mask_entry["mask_inner"] = res["mask_inner"]
            mask_entry["ring"] = res["ring"]
        masks.append(mask_entry)

    if not records:
        return empty_measurement_table(**blocks), masks
    return pd.DataFrame(records)[list(columns)], masks


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

    # No detector block: a box has a size but no sigma, and half a block is what
    # the `if k in res` assembly used to produce (ADR-0031).
    blocks = {
        "segmentation": True,
        "height": z_flat is not None,
        "geometry": z_flat is None,
    }
    columns = measurement_columns(**blocks)

    records: list[dict] = []
    masks: list[dict] = []

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

        res["particle_id"] = len(records)
        res["method"] = "sam2_boxes"
        records.append({name: res[name] for name in columns})

        mask_entry = {
            "x_px": res["x_px"],
            "y_px": res["y_px"],
            "mask": res["mask"],
            "mask_score": res["mask_score"],
            "box": box,
        }
        if "mask_inner" in res:
            mask_entry["mask_inner"] = res["mask_inner"]
            mask_entry["ring"] = res["ring"]
        masks.append(mask_entry)

    if not records:
        return empty_measurement_table(**blocks), masks
    return pd.DataFrame(records)[list(columns)], masks
