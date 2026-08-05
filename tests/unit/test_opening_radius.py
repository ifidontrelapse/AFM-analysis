"""Every radius reaching `disk()` is an integer, rounded up (D-10, ADR-0020).

`disk(8.5)` is an 18x18 structuring element: an even side, so there is no centre
pixel and the morphological opening is biased by half a pixel. Any integer radius
gives `2r+1`, which is always odd and always centred — so the only question was
which way to round, and B4 answered "up": a radius that is too small leaves
particles inside the estimated substrate, which is the error that corrupts every
height downstream.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.morphology import disk

from nanoscope.core.science.preprocessing import build_substrate_map, get_substrate_map
from nanoscope.core.science.preprocessing.substrate import _integer_radius, estimate_rough_radius


def _particles(size: int = 64) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size), dtype=np.float32)
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        z += 6.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 4.0**2))
    return z


@pytest.mark.parametrize("radius", [5.0, 5.5, 8.5, 11.9, 1.0, 0.4])
def test_the_structuring_element_always_has_a_centre_pixel(radius: float) -> None:
    """The property the whole decision exists for. Half-integer radii used to
    produce an even side — measured in the audit: 8.5 -> (18, 18)."""
    element = disk(_integer_radius(radius))
    assert element.shape[0] % 2 == 1
    assert element.shape[0] == element.shape[1]


def test_rounding_is_up_never_down() -> None:
    """Down would shrink the disk below the particle it has to step over."""
    assert _integer_radius(8.5) == 9
    assert _integer_radius(8.01) == 9
    assert _integer_radius(8.0) == 8


def test_the_rough_estimate_returns_the_int_it_promises() -> None:
    """It was annotated `-> int` and returned a float — the return-type lie the
    audit recorded alongside the centring bug."""
    radius = estimate_rough_radius(_particles(), 1.0, min_size_pixel=2)
    assert isinstance(radius, int)


def test_a_flat_image_falls_back_to_an_integer_too() -> None:
    """The other exit from the same function, which used to floor `width * 0.01`
    while the main exit did not round at all."""
    radius = estimate_rough_radius(np.zeros((256, 256), dtype=np.float32), 1.0, min_size_pixel=1)
    assert isinstance(radius, int)
    assert radius == 3  # ceil(256 * 0.01)


def test_a_fractional_manual_radius_is_reported_as_the_integer_used() -> None:
    """ADR-0014 made this branch return the radius it actually uses. Opening with
    9 and reporting 8.5 would put that lie back one field along."""
    _, _, opening_radius, _ = build_substrate_map(
        _particles(), pixel_size_nm=1.0, min_size_nm=1, manual_radius_px=8.5
    )
    assert opening_radius == 9
    assert isinstance(opening_radius, int)


def test_the_substrate_is_identical_to_the_one_the_integer_radius_gives() -> None:
    """`get_substrate_map` is the funnel every caller passes through, so the
    guard lives there rather than at each call site."""
    z = _particles()
    assert np.array_equal(get_substrate_map(z, 8.5), get_substrate_map(z, 9))
    assert not np.array_equal(get_substrate_map(z, 8.5), get_substrate_map(z, 8))
