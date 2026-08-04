"""
AFM nanoparticle analysis pipeline.

Typical usage:

    from src.pipeline import run_pipeline, PipelineConfig

    cfg    = PipelineConfig(detector="log", mode="segment")
    result = run_pipeline(pre, cfg, predictor)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.types import (
    Detection, PipelineConfig, PipelineResult,
    PreprocessingResult, MicroscopyData,
)
from src.detection import LogDetector, YoloDetector
from src.segmentation import run_sam2_from_blobs, run_sam2_from_boxes
from src.measure import measure_all_baseline

from nanoscope.application.capabilities import validate_request


def run_pipeline(
    data: PreprocessingResult | MicroscopyData,
    cfg: PipelineConfig,
    predictor=None,
) -> PipelineResult:
    """
    Run the full AFM/SEM/TEM nanoparticle analysis pipeline.

    Args:
        data:      PreprocessingResult (AFM) or MicroscopyData (SEM/TEM)
        cfg:       pipeline configuration
        predictor: initialised SAM2ImagePredictor — required when cfg.mode="segment".
                   If None and mode="segment", raises ValueError.

    Returns:
        PipelineResult
    """
    # ── Unpack modality-specific fields ──────────────────────────────────────
    if isinstance(data, PreprocessingResult):
        image        = data.z_result
        nm_per_pixel = data.pixel_size_nm
        z_flat       = data.z_flat
        sizes        = data.sizes
        modality     = "afm"
    else:
        image        = data.image
        nm_per_pixel = data.nm_per_pixel
        z_flat       = None
        sizes        = None
        modality     = data.modality

    # ── Validate before spending anything (M2-T10, audit D-14) ───────────────
    # Every rule about which (modality, detector, mode) combinations exist lives
    # in one module now, and it is consulted *here* — before a detector is
    # constructed. This block used to sit after inference, so AFM + YOLO +
    # baseline ran a full YOLO pass and then raised.
    validate_request(modality, cfg.detector, cfg.mode, has_predictor=predictor is not None)

    # ── Detection ────────────────────────────────────────────────────────────
    if cfg.detector == "log":
        detector   = LogDetector(
            overlap=cfg.log_overlap,
            percentile=cfg.log_percentile,
            threshold=cfg.log_threshold,
        )
        detections = detector.detect(image, nm_per_pixel, sizes=sizes)
        blobs      = detector.last_blobs

    elif cfg.detector == "yolo":
        detector   = YoloDetector(
            model_path=cfg.yolo_model_path,
            use_tiling=cfg.yolo_use_tiling,
            conf=cfg.yolo_conf,
        )
        detections = detector.detect(image, nm_per_pixel)
        blobs      = None

    else:  # pragma: no cover — validate_request rejected this before we got here
        raise ValueError(f"Unknown detector: {cfg.detector!r}")

    # ── Detect-only early exit ────────────────────────────────────────────────
    if cfg.mode == "detect":
        return PipelineResult(
            detections=detections, masks=[], measurements=pd.DataFrame(),
            pixel_size_nm=nm_per_pixel, detector_name=cfg.detector,
            mode=cfg.mode, modality=modality,
        )

    # ── Baseline mode (AFM only) ──────────────────────────────────────────────
    if cfg.mode == "baseline":
        measurements = measure_all_baseline(
            z_flat, image, blobs,
            outer_px=cfg.measure_outer_px,
            inner_erode_px=cfg.measure_inner_erode_px,
        )
        return PipelineResult(
            detections=detections, masks=[], measurements=measurements,
            pixel_size_nm=nm_per_pixel, detector_name=cfg.detector,
            mode=cfg.mode, modality=modality,
        )

    # ── Segment mode ──────────────────────────────────────────────────────────
    if cfg.detector == "log":
        measurements, masks = run_sam2_from_blobs(
            predictor, z_flat, image, blobs,
            nm_per_pixel=nm_per_pixel,
            outer_ring_px=cfg.sam2_outer_ring_px,
            inner_erode_px=cfg.sam2_inner_erode_px,
        )
    else:
        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        measurements, masks = run_sam2_from_boxes(
            predictor, z_flat, image, boxes,
            nm_per_pixel=nm_per_pixel,
            outer_ring_px=cfg.sam2_outer_ring_px,
            inner_erode_px=cfg.sam2_inner_erode_px,
        )

    return PipelineResult(
        detections=detections, masks=masks, measurements=measurements,
        pixel_size_nm=nm_per_pixel, detector_name=cfg.detector,
        mode=cfg.mode, modality=modality,
    )
