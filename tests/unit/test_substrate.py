"""`build_substrate_map`, and the manual-radius branch that never worked (D-01).

The characterization golden records that this branch used to raise, so the fix is
visible there too. These tests exist because the golden can only say *what*
changed — it cannot say the new behaviour is the intended one.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core.science.preprocessing import build_substrate_map


def _particles(size: int = 64, radius: float = 4.0) -> np.ndarray:
    """A flat field with four Gaussian bumps — enough for Otsu to find objects."""
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size), dtype=np.float32)
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        z += 10.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * radius**2))
    return z


class TestManualRadius:
    def test_the_manual_branch_returns_instead_of_raising(self) -> None:
        # Before M3-T01 this raised UnboundLocalError on 100% of calls: the
        # branch assigned `opening_radius` only in the `else`.
        substrate, z_above, _radius, sizes = build_substrate_map(
            _particles(), pixel_size_nm=1.0, min_size_nm=5, manual_radius_px=15
        )
        assert substrate.shape == (64, 64)
        assert z_above.shape == (64, 64)
        assert sizes["typical_radius_px"] > 0

    def test_it_reports_the_radius_it_was_given(self) -> None:
        """The returned radius is the one actually used, not a re-derived guess.

        `opening_radius` is documented as "the radius finally used". On the manual
        branch that is exactly the caller's value — it is what
        `get_substrate_map` was called with. Rounding it here would be a separate
        decision, and it is B4/M3-T09's.
        """
        for requested in (7, 15, 21):
            *_, opening_radius, _ = build_substrate_map(
                _particles(), pixel_size_nm=1.0, min_size_nm=5, manual_radius_px=requested
            )
            assert opening_radius == requested

    def test_a_different_radius_produces_a_different_substrate(self) -> None:
        # Guards against the fix being cosmetic: the radius must still reach the
        # morphological opening, not just the return value.
        z = _particles()
        _, above_small, _, _ = build_substrate_map(z, 1.0, 5, manual_radius_px=6)
        _, above_large, _, _ = build_substrate_map(z, 1.0, 5, manual_radius_px=20)
        assert not np.allclose(above_small, above_large)

    def test_the_automatic_branch_is_untouched(self) -> None:
        """M3-T01 must not have moved the path 100% of real callers use."""
        z = _particles()
        _, _, auto_radius, _ = build_substrate_map(z, pixel_size_nm=1.0)
        # The automatic radius is derived, floored at 5, and an int.
        assert isinstance(auto_radius, int)
        assert auto_radius >= 5

    def test_manual_and_automatic_agree_when_given_the_same_radius(self) -> None:
        z = _particles()
        _, above_auto, auto_radius, _ = build_substrate_map(z, pixel_size_nm=1.0)
        _, above_manual, manual_radius, _ = build_substrate_map(
            z, pixel_size_nm=1.0, manual_radius_px=auto_radius
        )
        assert manual_radius == auto_radius
        # Same radius, same opening, same result — the two branches differ only
        # in how the radius is chosen.
        np.testing.assert_allclose(above_manual, above_auto, rtol=1e-6)


def test_an_empty_image_still_raises_from_otsu() -> None:
    """The fix must not swallow the failure modes the golden already records."""
    with pytest.raises(ValueError, match="Otsu found no objects"):
        build_substrate_map(np.zeros((32, 32), np.float32), 1.0, 5, manual_radius_px=5)
