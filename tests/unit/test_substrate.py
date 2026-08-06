"""`build_substrate_map` and `estimate_radius_otsu`: D-01, and D-05/D-06.

The characterization golden records that this branch used to raise, so the fix is
visible there too. These tests exist because the golden can only say *what*
changed — it cannot say the new behaviour is the intended one.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core.science.preprocessing import build_substrate_map, estimate_radius_otsu


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
        # min_size_nm=1, not the default 5: at 1 nm/px the fixture's 4.7 px
        # particles are all below a 5 px floor, and since M3-T06 that is an
        # error rather than a silent nan. This test is about the radius, not
        # about the size filter.
        _, above_small, _, _ = build_substrate_map(z, 1.0, 1, manual_radius_px=6)
        _, above_large, _, _ = build_substrate_map(z, 1.0, 1, manual_radius_px=20)
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


def _two_sizes(size: int = 64) -> np.ndarray:
    """Two large bumps and two small ones — the audit's 4-objects-2-retained case."""
    ys, xs = np.mgrid[0:size, 0:size]
    z = np.zeros((size, size), dtype=np.float32)
    for (cy, cx), radius in (((16, 16), 6.0), ((16, 48), 6.0), ((48, 16), 1.5), ((48, 48), 1.5)):
        z += 10.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * radius**2))
    return z


class TestOtsuSizing:
    """D-05 and D-06 — what `estimate_radius_otsu` does when the size filter
    bites, and what it reports when it does not."""

    def test_filtering_everything_away_raises_instead_of_returning_nan(self) -> None:
        """The audit's own reproduction. `np.median([])` is `nan` with a
        RuntimeWarning, and the `nan` used to surface several calls later as
        "zero-size array to reduction operation minimum"."""
        with pytest.raises(ValueError, match="none with a radius of at least"):
            estimate_radius_otsu(_particles(), pixel_size_nm=1.0, min_size_nm=500)

    def test_the_error_names_the_parameter_its_value_and_the_largest_object(self) -> None:
        """PROJECT_RULES §3: an error names the offending parameter and its
        value. Here it also names what the filter was measured against, because
        "no particles" and "the threshold is 100x too high" look identical from
        the caller's side."""
        with pytest.raises(ValueError) as excinfo:
            estimate_radius_otsu(_particles(), pixel_size_nm=1.0, min_size_nm=500)
        message = str(excinfo.value)
        assert "min_size_nm=500" in message
        assert "the largest is" in message

    def test_n_objects_counts_what_survived_the_filter(self) -> None:
        """D-06: it used to report the pre-filter count, so a caller reading it
        as "particles found" over-counted — 4 reported against 2 retained in the
        audit's measurement, which is exactly what this reproduces."""
        z = _two_sizes()
        unfiltered = estimate_radius_otsu(z, 1.0, min_size_nm=0)
        filtered = estimate_radius_otsu(z, 1.0, min_size_nm=3)
        assert unfiltered["n_objects"] == 4
        assert filtered["n_objects"] == 2 == len(filtered["radii_px"])

    def test_n_objects_is_unchanged_when_the_filter_removes_nothing(self) -> None:
        """Nothing is filtered, so the old and new counts agree — which is why
        the golden moved on only some phantoms in M3-T06. Until M3-T02 this was
        also *the common case on real scans*, because D-04 floored the threshold
        to 0 there; since ADR-0024 it takes an explicit `min_size_nm=0`."""
        sizes = estimate_radius_otsu(_particles(), 1.0, min_size_nm=0)
        assert sizes["n_objects"] == len(sizes["radii_px"])


def _blocks(size: int = 64) -> np.ndarray:
    """Three objects with **exact** pixel areas, so their equivalent radii are
    arithmetic rather than an artefact of where a Gaussian's tail meets Otsu:

    | block | area px^2 | equivalent radius px |
    |---|---|---|
    | 8x8 | 64 | 4.514 |
    | 4x4 | 16 | 2.257 |
    | 1x1 | 1  | 0.564 |

    0.564 px is the smallest radius a labelling can produce, and it is the whole
    of D-04: at 6 nm/px that single pixel is 3.4 nm, below the 5 nm default and
    above the **0 px** the default used to be compared against.
    """
    z = np.zeros((size, size), dtype=np.float32)
    z[8:16, 8:16] = 10.0
    z[8:12, 40:44] = 10.0
    z[40, 40] = 10.0
    return z


class TestTheMinimumSizeIsPhysical:
    """D-04 (ADR-0024, decision B2). `int(min_size_nm / pixel_size_nm)` is 0 on
    **568 of the operator's 628 scans**, so the noise filter admitted every
    connected component — including single-pixel noise, whose radii then set the
    opening radius and the LoG sigma range. The threshold is now compared in
    nanometres, where it was always stated.

    The two cases below are the two regimes those 628 scans fall into, and each
    turns red if the `int()` comes back.
    """

    def test_a_coarse_scan_no_longer_admits_single_pixel_noise(self) -> None:
        """6 nm/px — the band holding 203 of the 628 real scans, where the old
        threshold floored to 0 *and* there was noise small enough for a working
        filter to remove. The single pixel is 3.4 nm against a 5 nm minimum."""
        sizes = estimate_radius_otsu(_blocks(), pixel_size_nm=6.0, min_size_nm=5)
        assert sizes["n_objects"] == 2  # 5 under `int(5 / 6.0) == 0`
        assert sizes["radii_nm"].min() >= 5.0

    def test_a_fine_scan_no_longer_quantises_the_threshold_down(self) -> None:
        """2 nm/px, where `int()` does not floor to zero and is still wrong: it
        truncates a 2.5 px threshold to 2 px, which admits the 4.5 nm block. The
        golden's `afm_sparse_low_snr` is this case — 75 objects became 17."""
        sizes = estimate_radius_otsu(_blocks(), pixel_size_nm=2.0, min_size_nm=5)
        assert sizes["n_objects"] == 1  # 2 under `int(5 / 2.0) == 2`
        assert sizes["radii_nm"].min() >= 5.0

    def test_an_unresolvable_minimum_removes_nothing_and_that_is_correct(self) -> None:
        """The third regime, 365 of the 628 scans: above 8.86 nm/px a single
        pixel is already wider than 5 nm, so nothing *can* be removed. The old
        code also removed nothing here — by accident, and the ADR's argument
        against a 1 px floor is that it would have removed everything."""
        sizes = estimate_radius_otsu(_blocks(), pixel_size_nm=9.77, min_size_nm=5)
        assert sizes["n_objects"] == 3
        assert sizes["radii_nm"].min() == pytest.approx(5.512, abs=1e-3)

    def test_the_threshold_is_the_stated_size_not_a_pixel_multiple(self) -> None:
        """The point of the decision. The same physical minimum keeps the same
        physical sizes at every scale: 4 nm keeps everything at least 4 nm
        across, whether one pixel is 1 nm or 9.77 — 1, 2, 2 and 3 of the three
        blocks respectively, and never the whole set for want of a threshold."""
        for pixel_size_nm in (1.0, 2.0, 6.0, 9.77):
            sizes = estimate_radius_otsu(_blocks(), pixel_size_nm, min_size_nm=4)
            expected = _blocks_radii_nm(pixel_size_nm) >= 4
            assert sizes["n_objects"] == int(expected.sum())
            assert sizes["radii_nm"].min() >= 4

    def test_the_error_now_speaks_nanometres(self) -> None:
        """The caller set a physical minimum, so the complaint has to be in the
        same units — PROJECT_RULES §3 again (ADR-0017 wrote this message)."""
        with pytest.raises(ValueError) as excinfo:
            estimate_radius_otsu(_particles(), pixel_size_nm=1.0, min_size_nm=500)
        assert "min_size_nm=500 nm" in str(excinfo.value)
        assert "px" not in str(excinfo.value)


def _blocks_radii_nm(pixel_size_nm: float) -> np.ndarray:
    """`_blocks`'s radii, computed from the areas rather than measured — the
    expectation has to come from somewhere other than the code under test."""
    areas = np.array([64.0, 16.0, 1.0])
    return np.sqrt(4 * areas / np.pi) / 2 * pixel_size_nm
