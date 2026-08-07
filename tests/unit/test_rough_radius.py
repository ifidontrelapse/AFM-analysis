"""A rough radius that lands below one pixel is not an estimate (B-061, ADR-0034).

`estimate_rough_radius` could return **0**, and `get_substrate_map(z, 0)` opens
with `disk(0)` — a single pixel — so the opening is the identity: the substrate
comes back equal to the image and `z_above` is zero everywhere. It looks like a
result.

The condition behind it is that `median + std` selected single-pixel noise
instead of particles, which is what the function's existing "too flat or too
noisy" branch is for. These tests pin the new route into that branch, and the
images that must not take it.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from nanoscope.core.science.preprocessing import (
    build_substrate_map,
    estimate_rough_radius,
    get_substrate_map,
)


def _noise(size: int = 128, seed: int = 0) -> np.ndarray:
    """Pure noise. Thresholding at `median + std` selects single pixels, so the
    median object area is 1 px and the radius estimate is worthless."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (size, size)).astype(np.float32)


def _particles(size: int = 128, radius: float = 6.0) -> np.ndarray:
    """Real particles: the estimate must come from them, not from a fallback."""
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size), dtype=np.float32)
    for cy, cx in ((32, 32), (32, 96), (96, 32), (96, 96)):
        z += 10.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * (radius / 1.5) ** 2))
    return z


class TestTheReproduction:
    def test_the_estimate_is_never_below_one_pixel(self) -> None:
        """It used to be exactly 0 on this input, with no scale to floor it."""
        radius = estimate_rough_radius(_noise(), pixel_size_nm=None, min_size_nm=5)

        assert radius >= 1

    def test_and_therefore_the_opening_is_never_the_identity(self) -> None:
        """The consequence that made a zero radius dangerous rather than merely
        wrong: `disk(0)` is one pixel, so `opening(z, disk(0)) == z` and the
        substrate is the image."""
        z = _noise()

        radius = estimate_rough_radius(z, pixel_size_nm=None, min_size_nm=5)
        substrate = get_substrate_map(z, radius)

        assert not np.array_equal(substrate, z.astype(np.float32))
        assert (z - substrate).max() > 0

    def test_a_zero_radius_would_still_be_an_identity_opening(self) -> None:
        """Pinned so the reason the guard exists cannot quietly stop being true.
        `get_substrate_map` still *accepts* 0 — M3-T13 deliberately left it
        legal — and this is what it does."""
        z = _noise()

        assert np.array_equal(get_substrate_map(z, 0), z.astype(np.float32))


class TestTheFallback:
    def test_it_is_one_percent_of_the_image_width(self) -> None:
        """The same fallback the empty case already used, reached by a second
        route rather than reimplemented."""
        assert estimate_rough_radius(_noise(size=128), None, min_size_nm=5) == 2  # ceil(1.28)
        assert estimate_rough_radius(_noise(size=256), None, min_size_nm=5) == 3  # ceil(2.56)

    def test_it_says_which_case_it_was(self, caplog: pytest.LogCaptureFixture) -> None:
        """Two routes into one fallback, and a reader needs to know which: an
        image with nothing in it and an image full of noise are different
        problems with the same answer."""
        with caplog.at_level(logging.WARNING):
            estimate_rough_radius(_noise(), None, min_size_nm=5)

        assert "sub-pixel" in caplog.text
        assert "that is noise, not a particle" in caplog.text
        assert "no objects found" not in caplog.text

    def test_the_empty_case_still_says_its_own_thing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            estimate_rough_radius(np.zeros((64, 64), dtype=np.float32), None, min_size_nm=5)

        assert "no objects found" in caplog.text
        assert "sub-pixel" not in caplog.text


class TestImagesWithParticlesAreUntouched:
    def test_the_estimate_comes_from_the_particles(self, caplog: pytest.LogCaptureFixture) -> None:
        """The four phantoms that do not move, in miniature: a real estimate is
        several pixels, so the new branch never sees them."""
        with caplog.at_level(logging.WARNING):
            radius = estimate_rough_radius(_particles(), pixel_size_nm=2.0, min_size_nm=5)

        assert radius > 1
        assert "sub-pixel" not in caplog.text

    def test_with_and_without_a_scale_agree_when_the_estimate_is_real(self) -> None:
        """The floor only ever mattered where the estimate was worthless. Where
        it is not, `pixel_size_nm=None` changes nothing about the radius."""
        z = _particles()

        assert estimate_rough_radius(z, 2.0, min_size_nm=5) == estimate_rough_radius(
            z, None, min_size_nm=5
        )


class TestWhatItCostTheSubstrate:
    def test_a_noisy_unscaled_run_no_longer_measures_an_unopened_map(self) -> None:
        """ADR-0025 recorded 17 objects → 3351 on this path and read it as the
        lost `min_size_nm` filter. The filter was the smaller half: with the
        rough opening restored, Otsu sees an opened map and the count falls by
        about four fifths.

        The scaled and unscaled runs still differ — the filter is still skipped
        — which is the claim M3-T20's test makes and this must not undo.
        """
        rng = np.random.default_rng(1)
        z = _particles() + rng.normal(0.0, 1.2, (128, 128)).astype(np.float32)

        scaled = build_substrate_map(z, 2.0)
        unscaled = build_substrate_map(z, None)

        assert unscaled[3]["n_objects"] > scaled[3]["n_objects"]
        assert not np.array_equal(unscaled[0], z.astype(np.float32))
