"""An unknown pixel scale is a supported state, not a TypeError (D-07, ADR-0019).

`MicroscopyData.nm_per_pixel` is `float | None` and `run_pipeline` hands that
value straight to the detector, so "scale unknown" reaches both detectors by the
ordinary route. Both used to multiply by it unconditionally:

    TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'

The rule these tests hold is the invariant D-07 names: no scale, no nanometres —
never zero, never a pixel count wearing nanometre units, never a crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core.entities import MicroscopyData, PipelineConfig
from nanoscope.core.science.detection import LogDetector
from nanoscope.core.science.detection.log import detect_particles

SIZES = {"radii_px": np.array([3.0, 5.0])}


def _bumps(size: int = 64) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size))
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        z += 6.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 4.0**2))
    return z


class TestLogDetector:
    def test_detect_particles_does_not_raise_without_a_scale(self) -> None:
        """The reproduction: this call was a TypeError."""
        blobs = detect_particles(_bumps(), None, SIZES, threshold=0.05)
        assert len(blobs) == 4

    def test_pixel_space_is_unaffected_by_the_missing_scale(self) -> None:
        """Only the nm column may differ — positions and sigmas are measured in
        pixels and do not need a scale to exist."""
        with_scale = detect_particles(_bumps(), 2.0, SIZES, threshold=0.05)
        without = detect_particles(_bumps(), None, SIZES, threshold=0.05)
        assert np.array_equal(with_scale[:, :3], without[:, :3])

    def test_the_radius_column_is_nan_not_a_pixel_count(self) -> None:
        """An ndarray column cannot hold None, so it holds NaN — and it must not
        hold `radius_px`, which is the failure mode the invariant forbids."""
        blobs = detect_particles(_bumps(), None, SIZES, threshold=0.05)
        assert np.isnan(blobs[:, 3]).all()

    def test_detections_report_none_rather_than_a_number(self) -> None:
        dets = LogDetector(threshold=0.05).detect(_bumps(), None, sizes=SIZES)
        assert len(dets) == 4
        assert all(d.radius_nm is None for d in dets)
        assert all(d.radius_px > 0 for d in dets)

    def test_a_known_scale_still_produces_nanometres(self) -> None:
        """The guard must not cost the working path anything."""
        dets = LogDetector(threshold=0.05).detect(_bumps(), 2.0, sizes=SIZES)
        assert all(d.radius_nm == pytest.approx(d.radius_px * 2.0) for d in dets)


class TestYoloDetector:
    """`_boxes_to_detections` is the site; it needs no weights, so it is
    testable without torch (and CI installs none)."""

    def test_boxes_without_a_scale_report_none(self) -> None:
        from nanoscope.infrastructure.models import YoloDetector

        dets = YoloDetector._boxes_to_detections(np.array([[10.0, 10.0, 30.0, 34.0]]), None)
        assert dets[0].radius_nm is None
        assert dets[0].radius_px == 10.0

    def test_boxes_with_a_scale_are_unchanged(self) -> None:
        from nanoscope.infrastructure.models import YoloDetector

        dets = YoloDetector._boxes_to_detections(np.array([[10.0, 10.0, 30.0, 34.0]]), 3.0)
        assert dets[0].radius_nm == pytest.approx(30.0)


def test_the_pipeline_path_that_reaches_it() -> None:
    """The route D-07 actually travels: SEM/TEM carries `nm_per_pixel: float |
    None`, and `run_pipeline` passes it to the detector without looking."""
    from nanoscope.application.use_cases import run_pipeline

    data = MicroscopyData(image=_bumps(), nm_per_pixel=None, modality="sem")
    # An explicit threshold, so that this test fails for D-07's reason and not
    # because the adaptive one found nothing in a four-bump toy image.
    result = run_pipeline(data, PipelineConfig(detector="log", mode="detect", log_threshold=0.05))

    assert result.pixel_size_nm is None
    assert result.detections
    assert all(d.radius_nm is None for d in result.detections)
