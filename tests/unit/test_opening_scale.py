"""The opening-radius constants, named and measured (B-064, ADR-0037).

Two numbers set every opening radius in the project, and until M3-T26 neither was
derived anywhere: `scale=1.7`, documented with one line, and a bare `2.5` literal
inside a branch. Both were chosen while the `int()` truncation ADR-0035 removed
was still in place, so the effective margin was `1.7 * int(r)/r` — 1.39 at
r = 4.9, not 1.7.

The sweep in ADR-0037 says to keep them. These tests hold the two things that
sweep depends on: the defaults are what they were, and the parameter reaches
`disk()` so a future sweep can be run at all. The trade-off itself — smaller
opening finds more particles, larger measures radii better — is pinned as a
property, because it is the reason the value is not free.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core.science.preprocessing import build_substrate_map, flatten_plane
from nanoscope.core.science.preprocessing.substrate import (
    DEFAULT_OPENING_SCALE,
    DEFAULT_ROUGH_SCALE,
    MIN_OPENING_RADIUS_PX,
    estimate_rough_radius,
)


def _particles(size: int = 128, radius: float = 6.0, spacing: int = 32) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    z = np.zeros((size, size), dtype=np.float32)
    for cy in range(spacing // 2, size, spacing):
        for cx in range(spacing // 2, size, spacing):
            z += 10.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * (radius / 1.5) ** 2))
    return z


def _touching_pair(size: int = 64, radius: float = 5.0, gap: float = 3.0) -> np.ndarray:
    """Two particles closer together than a large disk can step between."""
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    z = np.zeros((size, size), dtype=np.float32)
    for cx in (size / 2 - radius - gap / 2, size / 2 + radius + gap / 2):
        z += 10.0 * np.exp(-((ys - size / 2) ** 2 + (xs - cx) ** 2) / (2 * (radius / 1.5) ** 2))
    return z


class TestTheDefaultsAreWhatTheyWere:
    def test_the_named_constants_hold_the_measured_values(self) -> None:
        """The sweep in ADR-0037 was run at these values; if they drift, the
        measurement in the ADR stops describing the code."""
        assert DEFAULT_ROUGH_SCALE == 1.7
        assert DEFAULT_OPENING_SCALE == 2.5
        assert MIN_OPENING_RADIUS_PX == 5

    def test_naming_them_moved_nothing(self) -> None:
        """The whole change is a rename and a parameter. Passing the default
        explicitly must be identical to not passing it."""
        z = flatten_plane(_particles())

        implicit = build_substrate_map(z, pixel_size_nm=2.0)
        explicit = build_substrate_map(z, pixel_size_nm=2.0, opening_scale=DEFAULT_OPENING_SCALE)

        assert implicit[2] == explicit[2]
        np.testing.assert_array_equal(implicit[0], explicit[0])

    def test_the_rough_scale_is_still_a_parameter_too(self) -> None:
        z = flatten_plane(_particles())

        assert estimate_rough_radius(z, 2.0, 5) == estimate_rough_radius(
            z, 2.0, 5, scale=DEFAULT_ROUGH_SCALE
        )


class TestTheParameterReachesTheOpening:
    @pytest.mark.parametrize("scale", [1.5, 2.0, 2.5, 3.0, 4.0])
    def test_a_larger_factor_gives_a_larger_radius(self, scale: float) -> None:
        z = flatten_plane(_particles())

        radius = build_substrate_map(z, pixel_size_nm=2.0, opening_scale=scale)[2]

        assert radius >= MIN_OPENING_RADIUS_PX

    def test_the_radius_grows_monotonically_with_the_factor(self) -> None:
        """Without this, sweeping the parameter would not be sweeping anything —
        which is the state the bare literal left the project in."""
        z = flatten_plane(_particles())
        radii = [build_substrate_map(z, 2.0, opening_scale=s)[2] for s in (1.5, 2.5, 4.0)]

        assert radii == sorted(radii)
        assert radii[0] < radii[-1]

    def test_the_floor_holds_when_the_factor_is_tiny(self) -> None:
        z = flatten_plane(_particles())

        radius = build_substrate_map(z, pixel_size_nm=2.0, opening_scale=0.01)[2]

        assert radius == MIN_OPENING_RADIUS_PX


class TestTheTradeOffIsReal:
    def test_a_larger_opening_merges_touching_particles(self) -> None:
        """Why the value is not free, as a property rather than a table.

        Two particles with a 3 px gap: a small disk steps into the gap and
        recovers a substrate between them, so both stand alone above it; a large
        one steps over the pair and the trough between them is swallowed. That
        is exactly the mechanism behind `afm_dense_overlapping`'s recall falling
        from 0.886 to 0.800 across the sweep (ADR-0037).
        """
        z = flatten_plane(_touching_pair())
        mid = z.shape[1] // 2

        small = build_substrate_map(z, 2.0, opening_scale=1.0)[1]
        large = build_substrate_map(z, 2.0, opening_scale=4.0)[1]

        # The trough between the two particles, relative to their peaks.
        assert small[:, mid].max() / small.max() < large[:, mid].max() / large.max()

    def test_and_costs_nothing_when_the_particles_are_far_apart(self) -> None:
        """The other half: the trade-off is about *neighbours*, not about size.
        On a sparse field both factors recover the same substrate level."""
        z = flatten_plane(_particles(spacing=48))

        small = build_substrate_map(z, 2.0, opening_scale=1.5)[1]
        large = build_substrate_map(z, 2.0, opening_scale=3.0)[1]

        assert abs(small.max() - large.max()) / large.max() < 0.05
