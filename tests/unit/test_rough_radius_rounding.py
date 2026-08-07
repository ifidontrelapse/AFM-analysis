"""The rough estimate does not round its own radius (B-063, ADR-0035).

```python
radius_px = int(np.sqrt(median_area / np.pi))  # truncates, downward, silently
rough_radius = max(radius_px * scale, min_size_px)
return _integer_radius(rough_radius)  # the declared rounding, upward
```

Two roundings in three lines, in opposite directions, and only the second one is
documented. ADR-0020 ruled that `_integer_radius` is the one funnel and that a
radius rounds **up** — "a radius smaller than a particle recovers a substrate
containing the particle" — and the `scale` parameter's own docstring calls itself
a multiplier making the disk *safely larger*. The truncation contradicted both.

These tests pin the arithmetic against the contract, computed from the image the
function was given: **`ceil(sqrt(median_area / pi) * scale)`, once**. What the
change does to the phantoms is the golden's job; what it does to detection
quality is `detection_quality`'s.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import label, regionprops

from nanoscope.core.science.preprocessing import estimate_rough_radius

SCALE = 1.7


def _disc_field(radius: float, size: int = 256, spacing: int = 40) -> np.ndarray:
    """Flat-topped discs of a known radius, well above `median + std`.

    Discs rather than Gaussians: the estimate is driven by `median_area`, and a
    rasterised disc's area is within a pixel of `pi * r**2`, so the expected
    answer can be *computed* from the image rather than approximated.
    """
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size), dtype=np.float32)
    for cy in range(spacing // 2, size, spacing):
        for cx in range(spacing // 2, size, spacing):
            z[((ys - cy) ** 2 + (xs - cx) ** 2) <= radius**2] = 10.0
    return z


def _equivalent_radius(z: np.ndarray) -> float:
    """The unrounded radius the function derives, read from the same image.

    This is the *input* to the rounding, not a reimplementation of the science:
    the threshold and the median area are what the function itself selects, and
    the tests below assert only what happens to the number afterwards.
    """
    flat = z.flatten()
    props = regionprops(label(z > np.median(flat) + flat.std()))
    return float(np.sqrt(float(np.median([p.area for p in props])) / np.pi))


class TestOnlyOneRounding:
    @pytest.mark.parametrize("radius", [3.5, 4.2, 4.9, 5.4, 6.7, 8.4])
    def test_the_answer_is_the_contract_exactly(self, radius: float) -> None:
        """`ceil(equivalent_radius * scale)` — one rounding, at the end.

        With `int()` in place the disc whose equivalent radius is 6.60 was
        estimated from 6, and the `* 1.7` turned a 0.6 px truncation into a
        1.0 px shortfall: 11 instead of 12.
        """
        z = _disc_field(radius)

        got = estimate_rough_radius(z, pixel_size_nm=1.0, min_size_nm=0)

        assert got == int(np.ceil(_equivalent_radius(z) * SCALE))

    @pytest.mark.parametrize("radius", [4.2, 4.9, 5.4, 6.7, 8.4])
    def test_and_it_is_larger_than_what_truncation_gave(self, radius: float) -> None:
        """The mutant's answer, computed alongside, on the radii where the two
        genuinely differ — `int()` discards a fraction that the `* 1.7` then
        amplifies past the next integer."""
        z = _disc_field(radius)
        truncated = int(np.ceil(int(_equivalent_radius(z)) * SCALE))

        got = estimate_rough_radius(z, pixel_size_nm=1.0, min_size_nm=0)

        assert got > truncated  # ADR-0020's direction: never smaller

    def test_where_the_truncation_happened_to_agree_it_still_does(self) -> None:
        """Not every radius exposed the defect, and pretending otherwise would
        overstate it: at an equivalent radius of 3.432, `ceil(3 * 1.7)` and
        `ceil(3.432 * 1.7)` are both 6. The fraction was too small to cross an
        integer — which is exactly why this radius is the useful half of the
        collision pair below, and why the defect was hard to see."""
        z = _disc_field(3.5)
        equivalent = _equivalent_radius(z)

        assert int(np.ceil(int(equivalent) * SCALE)) == int(np.ceil(equivalent * SCALE)) == 6
        assert estimate_rough_radius(z, pixel_size_nm=1.0, min_size_nm=0) == 6

    def test_the_rounding_that_remains_is_upward(self) -> None:
        """ADR-0020, still in force at the one site that rounds."""
        z = _disc_field(5.0)

        got = estimate_rough_radius(z, pixel_size_nm=1.0, min_size_nm=0)

        assert got >= _equivalent_radius(z) * SCALE
        assert isinstance(got, int)


class TestTheEstimateTracksTheParticles:
    def test_two_particles_inside_one_truncation_step_no_longer_collide(self) -> None:
        """Constructed to fail under the defect: discs of radius 3.5 and 3.7
        have equivalent radii 3.432 and 3.785, so `int()` maps **both** to 3 and
        both estimates came out 6. They are different particles, and the disks
        that step over them are now 6 and 7."""
        small, large = _disc_field(3.5), _disc_field(3.7)
        assert int(_equivalent_radius(small)) == int(_equivalent_radius(large))  # the collision

        got_small = estimate_rough_radius(small, pixel_size_nm=1.0, min_size_nm=0)
        got_large = estimate_rough_radius(large, pixel_size_nm=1.0, min_size_nm=0)

        assert (got_small, got_large) == (6, 7)

    def test_a_larger_particle_gives_a_larger_radius(self) -> None:
        small = estimate_rough_radius(_disc_field(4.2), 1.0, min_size_nm=0)
        large = estimate_rough_radius(_disc_field(8.4), 1.0, min_size_nm=0)

        assert large > small


class TestWhatDidNotChange:
    def test_the_minimum_size_floor_still_applies(self) -> None:
        """`max(radius * scale, min_size_px)` is untouched: a physical minimum
        still floors the estimate (ADR-0024)."""
        floored = estimate_rough_radius(_disc_field(4.0), pixel_size_nm=1.0, min_size_nm=40)

        assert floored == 40  # min_size_px = 40 / 1.0, well above 4 * 1.7

    def test_a_sub_pixel_estimate_still_falls_back(self) -> None:
        """M3-T23's guard survives, and reads more naturally now: the estimate it
        rejects is a genuine fraction rather than a truncated zero."""
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 1.0, (128, 128)).astype(np.float32)

        assert estimate_rough_radius(noise, pixel_size_nm=None, min_size_nm=5) == 2

    def test_an_empty_image_still_falls_back(self) -> None:
        assert estimate_rough_radius(np.zeros((256, 256), dtype=np.float32), None, 5) == 3
