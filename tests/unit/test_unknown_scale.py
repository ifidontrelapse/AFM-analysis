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
from nanoscope.core.science.preprocessing import build_substrate_map
from nanoscope.core.science.preprocessing.substrate import estimate_rough_radius

SIZES = {"radii_px": np.array([3.0, 5.0])}


def _two_sizes(size: int = 64) -> np.ndarray:
    """Two 6 px bumps and two 1.5 px ones, so a threshold can sit between them —
    `_bumps` is four identical particles, which a filter can only keep or empty."""
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size))
    for (cy, cx), radius in (((16, 16), 6.0), ((16, 48), 6.0), ((48, 16), 1.5), ((48, 48), 1.5)):
        z += 10.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * radius**2))
    return z


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


class TestPreprocessingWithoutAScale:
    """M3-T20 / ADR-0025 — the AFM half of the same invariant.

    Until this task the state was unreachable from `load_afm`, because the npy
    branch fabricated `1.0` nm/px. With the fabrication gone, `None` reaches
    `build_substrate_map`, which needs the scale for three things: `radii_nm`,
    the `min_size_nm` filter, and `estimate_rough_radius`'s floor. None of the
    three can be expressed without it, and the substrate itself needs none of
    them — which is what these tests pin.
    """

    def test_the_substrate_is_identical_where_the_filter_removed_nothing(self) -> None:
        """The opening is pixel-space arithmetic from end to end, so losing the
        scale costs nothing *when the filter was not doing anything anyway* — 4
        bumps of ~5 px radius at 2 nm/px are all far over the 5 nm default. Four
        of the five golden AFM phantoms are in this case; the fifth is the next
        test, and the difference between them is the honest scope of the claim."""
        z = _bumps()
        scaled = build_substrate_map(z, 2.0)
        unscaled = build_substrate_map(z, None)

        np.testing.assert_array_equal(scaled[0], unscaled[0])  # substrate
        np.testing.assert_array_equal(scaled[1], unscaled[1])  # z_above
        assert scaled[2] == unscaled[2]  # opening radius
        np.testing.assert_array_equal(scaled[3]["radii_px"], unscaled[3]["radii_px"])

    def test_no_scale_is_exactly_no_minimum_size(self) -> None:
        """The precise statement of what is skipped: an unscaled run equals a
        scaled run with `min_size_nm=0`, in every pixel-space field, including
        the rough-radius floor. Nothing else about the estimate changes."""
        z = _two_sizes()
        unscaled = build_substrate_map(z, None)
        no_minimum = build_substrate_map(z, 2.0, min_size_nm=0)

        np.testing.assert_array_equal(unscaled[0], no_minimum[0])
        assert unscaled[2] == no_minimum[2]
        assert unscaled[3]["n_objects"] == no_minimum[3]["n_objects"]
        np.testing.assert_array_equal(unscaled[3]["radii_px"], no_minimum[3]["radii_px"])

    def test_and_that_costs_the_substrate_on_a_noisy_scan(self) -> None:
        """Which is not free. Otsu on a noisy image finds objects the filter
        exists to remove; unfiltered, their radii set `typical_radius_px`, which
        sets the opening radius, so the substrate itself differs from the scaled
        run. The golden measures it on `afm_sparse_low_snr`: **17 objects become
        3351**, the typical radius falls 2.99 px → 0.80, and the opening radius
        goes 8 → 5. This is D-04's mechanism arriving by a different road, and it
        is why the skip is warned about rather than treated as equivalent."""
        rng = np.random.default_rng(0)
        z = _two_sizes() + rng.normal(0.0, 1.2, (64, 64))

        scaled = build_substrate_map(z, 2.0)
        unscaled = build_substrate_map(z, None)

        assert unscaled[3]["n_objects"] != scaled[3]["n_objects"]
        assert unscaled[2] != scaled[2]  # a different opening radius
        assert not np.array_equal(scaled[0], unscaled[0])

    def test_the_nanometre_fields_are_none_not_a_pixel_count(self) -> None:
        """D-07's invariant, stated on this dict: absent, never the pixel value
        wearing nanometre units — which is what `pixel_size_nm or 1.0` produced
        for every npy scan before ADR-0025."""
        sizes = build_substrate_map(_bumps(), None)[3]

        assert sizes["radii_nm"] is None
        assert sizes["typical_radius_nm"] is None
        assert sizes["typical_radius_px"] > 0
        assert sizes["n_objects"] == len(sizes["radii_px"])

    def test_the_dropped_size_filter_is_warned_about(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Skipping the filter silently would be D-04 again — a noise filter
        that is off and says nothing. It says something."""
        import logging

        logger_name = "nanoscope.core.science.preprocessing.substrate"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            build_substrate_map(_bumps(), None)

        assert "no physical scale" in caplog.text
        assert "5" in caplog.text  # the minimum it could not apply, named

    def test_no_object_is_filtered_away_without_a_scale(self) -> None:
        """The consequence, stated as a test rather than left to the log: every
        object Otsu found survives, because no physical threshold exists to
        remove it. The same image with a scale, and a minimum between the two
        particle sizes, keeps half of them."""
        z = _two_sizes()
        unscaled = build_substrate_map(z, None)[3]
        scaled = build_substrate_map(z, 2.0, min_size_nm=8)[3]

        assert unscaled["n_objects"] == 4
        assert scaled["n_objects"] == 2

    def test_the_rough_radius_floor_is_dropped_not_reinterpreted(self) -> None:
        """`min_size_nm` must not be silently read as pixels when the scale is
        missing — that is the unit confusion ADR-0024 deleted. With no scale the
        floor is 0 px, so a huge `min_size_nm` cannot inflate the radius."""
        flat = np.zeros((256, 256), dtype=np.float32)
        assert estimate_rough_radius(flat, None, min_size_nm=200) == 3  # ceil(256 * 0.01)
        assert estimate_rough_radius(flat, 1.0, min_size_nm=200) == 200

    def test_the_route_the_defect_actually_travels(self, tmp_path) -> None:
        """`run_preprocessing` on an npy with no metadata — the whole of M3-T20
        end to end. It used to return `pixel_size_nm=1.0` and a `scan_size_nm`
        equal to the row count, and every `_nm` downstream was a pixel count."""
        from nanoscope.application.use_cases import run_preprocessing

        path = tmp_path / "scan.npy"
        np.save(path, _two_sizes().astype(np.float32))

        result = run_preprocessing(path, fmt="npy")

        assert result.pixel_size_nm is None
        assert result.scan_size_nm is None
        assert result.sizes["typical_radius_nm"] is None
        assert result.sizes["typical_radius_px"] > 0
        assert result.z_result.shape == (64, 64)
