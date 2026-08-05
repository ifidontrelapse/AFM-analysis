"""`YoloDetector._prepare_image` — normalise first, cast second (D-03, ADR-0015).

Preparation is the only part of the YOLO path inside the gate: inference needs
weights and a GPU, and PROJECT_RULES §6 keeps it out. So everything that can be
asserted about the detector's input is asserted here.

Each test states a property of the *mapping* height → grey level. Under the old
cast-then-normalise order every one of them fails, which is the point: they are
written against the defect, not against the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.infrastructure.models import YoloDetector


@pytest.fixture
def det() -> YoloDetector:
    # No weights are touched: _prepare_image is pure image arithmetic, and the
    # constructor only stores parameters.
    return YoloDetector(yolo_size=64)


def _grey(det: YoloDetector, z: np.ndarray) -> np.ndarray:
    """The single grey channel the three RGB channels are copies of."""
    img = det._prepare_image(z)
    assert img.shape == (det.yolo_size, det.yolo_size, 3)
    assert img.dtype == np.uint8
    assert (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 0] == img[:, :, 2]).all()
    return img[:, :, 0]


def _ramp(size: int, lo: float, hi: float) -> np.ndarray:
    """A left-to-right height ramp: strictly increasing along x, constant in y."""
    return np.tile(np.linspace(lo, hi, size, dtype=np.float64), (size, 1))


class TestDynamicRange:
    def test_every_distinct_height_gets_its_own_grey_level(self, det: YoloDetector) -> None:
        """0 to 20 nm is an ordinary height map. A 64-step ramp has 64 distinct
        heights and must come out as 64 distinct levels spanning the full range;
        cast-first delivered 21 of them, because it kept only the integers."""
        levels = np.unique(_grey(det, _ramp(64, 0.0, 20.0)))
        assert levels.size == 64
        assert levels.min() == 0
        assert levels.max() == 255

    def test_a_sub_unit_range_survives(self, det: YoloDetector) -> None:
        """The worst case: a map whose whole range is smaller than one grey level.

        Cast-first collapses 0 to 0.8 nm to a single value — a uniform image, zero
        detections, no error.
        """
        assert np.unique(_grey(det, _ramp(64, 0.0, 0.8))).size == 64


class TestMonotonicity:
    def test_taller_is_never_brighter_after_inversion(self, det: YoloDetector) -> None:
        """The output is inverted, so height must map to a non-increasing grey."""
        row = _grey(det, _ramp(64, -5.0, 45.0))[0]
        assert (np.diff(row.astype(int)) <= 0).all()

    def test_a_range_beyond_255_does_not_wrap(self, det: YoloDetector) -> None:
        """`uint8(260) == 4`: cast-first turned the tallest particle into the
        darkest pixel. The audit's probe, as a map."""
        row = _grey(det, _ramp(64, -10.0, 300.0))[0]
        assert (np.diff(row.astype(int)) <= 0).all()
        assert row[0] == 255 and row[-1] == 0


class TestScaleInvariance:
    def test_the_result_depends_on_shape_not_on_units(self, det: YoloDetector) -> None:
        """Min-max normalisation makes preparation invariant under z → a·z + b
        for a > 0. Reporting the same scan in ångström must not change what YOLO
        sees; under cast-first it changes everything.
        """
        z = _ramp(64, 0.0, 12.0)
        base = _grey(det, z)
        assert np.abs(_grey(det, 10.0 * z + 100.0).astype(int) - base.astype(int)).max() <= 1


class TestDegenerate:
    def test_a_constant_map_is_uniform_and_does_not_raise(self, det: YoloDetector) -> None:
        """max == min, so there is no range to stretch. Unchanged by this fix,
        recorded because §6 requires degenerate inputs to have a stated answer."""
        assert np.unique(_grey(det, np.full((64, 64), 3.7))).size == 1
