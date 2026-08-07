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

from nanoscope.core.errors import InvalidImageError
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
        ],
        ids=["zeros", "constant_negative", "negative_structure", "positive"],
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

    def test_a_nan_map_is_refused_before_the_maximum_is_taken(self) -> None:
        """One nan pixel used to become an entirely nan image, via the division
        by `z_max`. ADR-0018 caught it with `not z_max > 0` and answered "no
        particles"; **ADR-0030 refuses the map instead**, and says how many
        values were not finite.

        The change is deliberate and narrow. A flat or negative map is valid
        data with nothing in it, and still gets ADR-0018's answer — that is the
        test three lines up. A map with a nan in it is not data this library can
        work with, and "no particles found" was the wrong sentence for it."""
        z = _bumps()
        z[10, 10] = np.nan

        with pytest.raises(InvalidImageError, match="1 nan"):
            detect_particles(z, 1.0, SIZES)

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
