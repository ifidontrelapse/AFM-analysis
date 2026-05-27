"""
AFM nanoparticle analysis pipeline.

Typical usage:

    from src.pipeline import run_pipeline, PipelineConfig

    cfg    = PipelineConfig(detector="log", mode="segment")
    result = run_pipeline(z_flat, z_result, pixel_size_nm, cfg)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.types import Detection, PipelineConfig, PipelineResult, PreprocessingResult
from src.detection import LogDetector, YoloDetector
from src.segmentation import run_sam2_from_blobs, run_sam2_from_boxes
from src.measure import measure_all_baseline


def run_pipeline(
    z_flat: np.ndarray,
    z_result: np.ndarray,
    pixel_size_nm: float,
    cfg: PipelineConfig,
    sizes: dict | None = None,
    predictor=None,
) -> PipelineResult:
    """
    Run the full AFM nanoparticle analysis pipeline.

    Args:
        z_flat:         flattened Z-map (nm)
        z_result:       z_flat - substrate (particles above substrate)
        pixel_size_nm:  nm/pixel
        cfg:            pipeline configuration
        sizes:          dict from estimate_radius_otsu — required for LogDetector.
                        If None, estimated automatically.
        predictor:      initialised SAM2ImagePredictor — required when cfg.mode="segment".
                        If None and mode="segment", raises ValueError.

    Returns:
        PipelineResult
    """
    # ── Detection ────────────────────────────────────────────────────────────
    if cfg.detector == "log":
        detector   = LogDetector(
            overlap=cfg.log_overlap,
            percentile=cfg.log_percentile,
            threshold=cfg.log_threshold,
        )
        detections = detector.detect(z_result, pixel_size_nm, sizes=sizes)
        blobs      = detector.last_blobs   # (N, 4) for SAM2 / measure_all_baseline

    elif cfg.detector == "yolo":
        detector   = YoloDetector(
            model_path=cfg.yolo_model_path,
            use_tiling=cfg.yolo_use_tiling,
            conf=cfg.yolo_conf,
        )
        detections = detector.detect(z_result, pixel_size_nm)
        blobs      = None   # YOLO uses boxes, not blobs

    else:
        raise ValueError(f"Unknown detector: {cfg.detector!r}")

    # ── Early exit ───────────────────────────────────────────────────────────
    if cfg.mode == "detect":
        return PipelineResult(
            detections=detections,
            masks=[],
            measurements=pd.DataFrame(),
            pixel_size_nm=pixel_size_nm,
            detector_name=cfg.detector,
            mode=cfg.mode,
        )

    # ── Baseline mode (LoG + circular masks, no SAM2) ────────────────────────
    if cfg.mode == "baseline":
        if cfg.detector != "log":
            raise ValueError("mode='baseline' is only supported with detector='log'")
        measurements = measure_all_baseline(
            z_flat, z_result, blobs,
            outer_px=cfg.measure_outer_px,
            inner_erode_px=cfg.measure_inner_erode_px,
        )
        return PipelineResult(
            detections=detections,
            masks=[],
            measurements=measurements,
            pixel_size_nm=pixel_size_nm,
            detector_name=cfg.detector,
            mode=cfg.mode,
        )

    # ── Segmentation ─────────────────────────────────────────────────────────
    if predictor is None:
        raise ValueError("predictor must be provided when mode='segment'")

    if cfg.detector == "log":
        measurements, masks = run_sam2_from_blobs(
            predictor, z_flat, z_result, blobs,
            pixel_size_nm=pixel_size_nm,
            outer_ring_px=cfg.sam2_outer_ring_px,
            inner_erode_px=cfg.sam2_inner_erode_px,
        )
    else:
        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        measurements, masks = run_sam2_from_boxes(
            predictor, z_flat, z_result, boxes,
            outer_ring_px=cfg.sam2_outer_ring_px,
            inner_erode_px=cfg.sam2_inner_erode_px,
        )

    return PipelineResult(
        detections=detections,
        masks=masks,
        measurements=measurements,
        pixel_size_nm=pixel_size_nm,
        detector_name=cfg.detector,
        mode=cfg.mode,
    )


def run_full_pipeline(
    pre: PreprocessingResult,
    cfg: PipelineConfig,
    predictor=None,
) -> PipelineResult:
    """
    Convenience wrapper: accepts a PreprocessingResult directly.

    Equivalent to calling run_pipeline() with the individual fields unpacked.
    This is the standard production call path.

    Args:
        pre:       output of run_preprocessing()
        cfg:       pipeline configuration
        predictor: initialised SAM2ImagePredictor — required when cfg.mode="segment"
    """
    return run_pipeline(
        z_flat=pre.z_flat,
        z_result=pre.z_result,
        pixel_size_nm=pre.pixel_size_nm,
        cfg=cfg,
        sizes=pre.sizes,
        predictor=predictor,
    )
