"""LoG normalisation against a non-positive maximum (D-11, ADR-0018).

`z_norm = z_above / z_above.max()` is the line. On a flat map it is `0/0` and
every pixel becomes `nan`; on a map that is negative everywhere it flips the
topography, so the substrate comes out brighter than the peaks. Neither case
raised — `blob_log` simply found nothing and the code blamed the threshold.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from nanoscope.core.science.detection.log import (
    DEFAULT_THRESHOLD,
    detect_particles,
    estimate_log_threshold_adaptive,
)

SIZES = {"radii_px": np.array([3.0, 5.0])}
PARAMS = {"min_sigma": 1.0, "max_sigma": 8.0}


def _bumps(offset: float = 0.0, size: int = 64) -> np.ndarray:
    """Four Gaussian caps, optionally pushed below zero by `offset`."""
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size))
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        z += 6.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 4.0**2))
    return z + offset


class TestThresholdStaysDimensionless:
    """The adaptive threshold is compared against a normalised image, so it is
    meaningless outside (0, 1]."""

    def test_a_negative_map_no_longer_yields_a_threshold_above_one(self) -> None:
        """The measured defect: dividing by a maximum of -4 produced **2.4997**,
        a threshold no normalised response can ever exceed, so the detector
        silently found nothing."""
        assert estimate_log_threshold_adaptive(_bumps(-10.0), PARAMS) == DEFAULT_THRESHOLD

    def test_a_flat_map_does_not_divide_by_zero(self) -> None:
        assert estimate_log_threshold_adaptive(np.zeros((64, 64)), PARAMS) == DEFAULT_THRESHOLD

    @pytest.mark.parametrize(
        "z",
        [
            np.zeros((64, 64)),
            np.full((64, 64), -5.0),
            _bumps(-10.0),
            _bumps(0.0),
            np.where(np.arange(64 * 64).reshape(64, 64) == 100, np.nan, 1.0),
        ],
        ids=["zeros", "constant_negative", "negative_structure", "positive", "with_nan"],
    )
    def test_the_threshold_is_always_in_the_unit_interval(self, z: np.ndarray) -> None:
        threshold = estimate_log_threshold_adaptive(z, PARAMS)
        assert 0.0 < threshold <= 1.0


class TestDetectParticles:
    def test_a_flat_map_returns_no_particles_and_says_why(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """D-11's real cost was the diagnosis: a numerical failure was reported
        as "try lowering the threshold", which sends the operator to tune a knob
        that cannot help."""
        with caplog.at_level(logging.WARNING):
            blobs = detect_particles(np.zeros((64, 64)), 1.0, SIZES)
        assert blobs.shape == (0, 4)
        assert "no positive signal above the substrate" in caplog.text
        assert "try lowering the threshold" not in caplog.text

    def test_a_nan_maximum_is_caught_rather_than_propagated(self) -> None:
        """`nan > 0` is False, so `not z_max > 0` catches it. The old division
        turned one nan pixel into an entirely nan image."""
        z = _bumps()
        z[10, 10] = np.nan
        assert detect_particles(z, 1.0, SIZES).shape == (0, 4)

    def test_the_sizes_argument_is_still_validated_first(self) -> None:
        """The guard sits after `estimate_log_params`, deliberately: a caller
        passing an unusable `sizes` should hear about that, not about the
        image."""
        with pytest.raises((KeyError, ValueError, IndexError)):
            detect_particles(np.zeros((64, 64)), 1.0, {"radii_px": np.array([])})

    def test_a_normal_map_is_untouched(self) -> None:
        """The guard must not cost the working path anything."""
        blobs = detect_particles(_bumps(), 1.0, SIZES, threshold=0.05)
        assert len(blobs) == 4
