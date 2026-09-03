"""Detect, then measure — the one orchestration the project has today.

Moved from `src/pipeline.py` in M2-T15. It is `application` rather than `core`
because it *coordinates*: it picks a detector, decides whether to measure heights
or run segmentation, and assembles the result. The science it calls stays in
`core.science`; the detector and segmenter it constructs live in
`infrastructure.models`.

That makes this the one place in the project where an outer layer is imported by
name rather than through a port. `Detector` (M2-T08) is the port that will remove
the `if/elif` here; the swap belongs to M4, where a container exists to do the
choosing.
"""

from __future__ import annotations

import numpy as np

from nanoscope.application.capabilities import validate_request
from nanoscope.core.entities import (
    MicroscopyData,
    PipelineConfig,
    PipelineResult,
    PreprocessingResult,
)
from nanoscope.core.errors import InvalidInputError, UnsupportedRequestError
from nanoscope.core.science.detection import LogDetector
from nanoscope.core.science.measurement import measure_all_baseline
from nanoscope.core.science.measurement.schema import blocks_for, empty_measurement_table
from nanoscope.core.values import default_polarity
from nanoscope.infrastructure.models import (
    YoloDetector,
    run_sam2_from_blobs,
    run_sam2_from_boxes,
)


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
    # The audit's first row: `run_pipeline("not-data", cfg)` answered with
    # `AttributeError: 'str' object has no attribute 'image'` — the name of a
    # field on the class the caller did *not* pass (D-15, ADR-0030).
    if not isinstance(data, (PreprocessingResult, MicroscopyData)):
        raise InvalidInputError(
            f"data must be a PreprocessingResult (AFM) or a MicroscopyData (SEM/TEM), "
            f"got {type(data).__name__}."
        )

    # ── Unpack modality-specific fields ──────────────────────────────────────
    # Annotated, not inferred: SEM/TEM carries `nm_per_pixel: float | None`, and
    # without this the variable takes the AFM branch's `float` and mypy reports
    # the SEM/TEM assignment as the error. That report *was* D-07 (M3-T11).
    nm_per_pixel: float | None
    if isinstance(data, PreprocessingResult):
        image = data.z_result
        nm_per_pixel = data.pixel_size_nm
        z_flat = data.z_flat
        sizes = data.sizes
        modality = "afm"
    else:
        image = data.image
        nm_per_pixel = data.nm_per_pixel
        z_flat = None
        sizes = None
        modality = data.modality

    # ── Validate before spending anything (M2-T10, audit D-14) ───────────────
    # Every rule about which (modality, detector, mode) combinations exist lives
    # in one module now, and it is consulted *here* — before a detector is
    # constructed. This block used to sit after inference, so AFM + YOLO +
    # baseline ran a full YOLO pass and then raised.
    validate_request(modality, cfg.detector, cfg.mode, has_predictor=predictor is not None)

    # Which way the particles read. Configured, not detected: an explicit
    # `cfg.polarity` wins, otherwise the modality's convention (ADR-0023 / B3).
    polarity = cfg.polarity or default_polarity(modality)

    # ── Detection ────────────────────────────────────────────────────────────
    if cfg.detector == "log":
        detector = LogDetector(
            overlap=cfg.log_overlap,
            percentile=cfg.log_percentile,
            threshold=cfg.log_threshold,
            polarity=polarity,
        )
        detections = detector.detect(image, nm_per_pixel, sizes=sizes)
        blobs = detector.last_blobs

    elif cfg.detector == "yolo":
        if not cfg.yolo_model_path:
            # Refused **before the detector is constructed**, which is where
            # M2-T10 put this class of refusal for D-14's reason: an impossible
            # request should cost milliseconds, not a preprocessing pass and
            # then a traceback. Until M8-T06 this could not happen, because the
            # field defaulted to a path — and that was the defect (W10).
            raise UnsupportedRequestError(
                "detector='yolo' names no weights. Register a model in this project "
                "and make it the active one, or pass yolo_model_path explicitly"
            )
        detector = YoloDetector(
            model_path=cfg.yolo_model_path,
            use_tiling=cfg.yolo_use_tiling,
            conf=cfg.yolo_conf,
            polarity=polarity,
        )
        detections = detector.detect(image, nm_per_pixel)
        blobs = None

    else:  # pragma: no cover — validate_request rejected this before we got here
        raise UnsupportedRequestError(f"Unknown detector: {cfg.detector!r}")

    # ── Detect-only early exit ────────────────────────────────────────────────
    if cfg.mode == "detect":
        return PipelineResult(
            detections=detections,
            masks=[],
            # Nothing was measured, by design — but "nothing" still has a shape.
            # `pd.DataFrame()` has zero columns, which is D-08 again in the one
            # place ADR-0027 left open on purpose: the schema here depends on the
            # modality, and that was M3-T14's decision to make (ADR-0031).
            # No `segmentation` block: this is the detect branch, so nothing
            # segmented anything. mypy said so first — `cfg.mode == "segment"`
            # here is a comparison that can only be False.
            measurements=empty_measurement_table(**blocks_for(modality)),
            pixel_size_nm=nm_per_pixel,
            detector_name=cfg.detector,
            mode=cfg.mode,
            modality=modality,
        )

    # ── Baseline mode (AFM only) ──────────────────────────────────────────────
    if cfg.mode == "baseline":
        measurements = measure_all_baseline(
            z_flat,
            image,
            blobs,
            outer_px=cfg.measure_outer_px,
            inner_erode_px=cfg.measure_inner_erode_px,
        )
        return PipelineResult(
            detections=detections,
            masks=[],
            measurements=measurements,
            pixel_size_nm=nm_per_pixel,
            detector_name=cfg.detector,
            mode=cfg.mode,
            modality=modality,
        )

    # ── Segment mode ──────────────────────────────────────────────────────────
    if cfg.detector == "log":
        measurements, masks = run_sam2_from_blobs(
            predictor,
            z_flat,
            image,
            blobs,
            nm_per_pixel=nm_per_pixel,
            outer_ring_px=cfg.sam2_outer_ring_px,
            inner_erode_px=cfg.sam2_inner_erode_px,
        )
    else:
        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        measurements, masks = run_sam2_from_boxes(
            predictor,
            z_flat,
            image,
            boxes,
            nm_per_pixel=nm_per_pixel,
            outer_ring_px=cfg.sam2_outer_ring_px,
            inner_erode_px=cfg.sam2_inner_erode_px,
        )

    return PipelineResult(
        detections=detections,
        masks=masks,
        measurements=measurements,
        pixel_size_nm=nm_per_pixel,
        detector_name=cfg.detector,
        mode=cfg.mode,
        modality=modality,
    )
