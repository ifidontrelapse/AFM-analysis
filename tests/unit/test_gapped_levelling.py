"""Levelling can fit around a gap instead of refusing the scan (B-060, ADR-0036).

M3-T13 made a non-finite value a rejection — the honest reading of what the code
already did, since `flatten_plane` had always refused NaN through `scipy.lstsq`.
It was never the best behaviour available: **a dropped scan line is a real
artefact**, and an AFM that loses feedback for two lines produces two rows of NaN
and four thousand good ones.

`allow_gaps=True` fits over the finite pixels and leaves the gap absent. It is
opt-in, and these tests hold both halves of that: the capability works, and the
default is still ADR-0030's contract.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from nanoscope.core.errors import InvalidImageError
from nanoscope.core.science.preprocessing import flatten_lines, flatten_plane

GAP_ROWS = (30, 31)


def _scene(size: int = 64) -> np.ndarray:
    """A tilted scan with four particles — no gap."""
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    z = 0.05 * xs + 0.03 * ys
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        z += 8.0 * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 4.0**2))
    return z


def _with_gap(rows: tuple[int, ...] = GAP_ROWS) -> np.ndarray:
    z = _scene()
    z[list(rows), :] = np.nan
    return z


class TestTheDefaultIsUnchanged:
    def test_a_gapped_map_is_still_refused(self) -> None:
        """ADR-0030's contract, still enforced, still with its own message."""
        with pytest.raises(InvalidImageError, match="not finite"):
            flatten_plane(_with_gap())

        with pytest.raises(InvalidImageError, match="not finite"):
            flatten_lines(_with_gap())

    def test_an_intact_map_levels_identically_either_way(self) -> None:
        """`allow_gaps` must be free when there are no gaps, or the flag would
        be a second implementation of levelling rather than one behaviour."""
        z = _scene()

        np.testing.assert_array_equal(flatten_plane(z), flatten_plane(z, allow_gaps=True))
        np.testing.assert_array_equal(flatten_lines(z), flatten_lines(z, allow_gaps=True))


class TestThePlaneFitsAroundTheGap:
    def test_it_recovers_what_the_ungapped_scan_would_have_given(self) -> None:
        """The measurement the decision rests on: masked levelling of a gapped
        scan agrees with levelling the same scan intact, to 0.03 nm on a map
        whose particles stand 8 nm tall."""
        intact = flatten_plane(_scene())

        levelled = flatten_plane(_with_gap(), allow_gaps=True)

        ok = np.isfinite(levelled)
        assert np.abs(levelled[ok] - intact[ok]).max() < 0.05

    def test_and_beats_filling_the_gap_with_zeros(self) -> None:
        """The tempting wrong fix, measured rather than dismissed. Zero-filling
        does not add noise — it tells the fit that the sample dips to zero along
        those lines, so it biases the tilt itself."""
        gapped = _with_gap()
        intact = flatten_plane(_scene())

        masked = flatten_plane(gapped, allow_gaps=True)
        filled = flatten_plane(np.nan_to_num(gapped, nan=0.0))

        ok = np.isfinite(masked)
        masked_err = np.abs(masked[ok] - intact[ok]).max()
        filled_err = np.abs(filled[ok] - intact[ok]).max()
        assert masked_err < filled_err / 3

    def test_the_gap_stays_absent(self) -> None:
        """Not filled, not interpolated: an interpolated value is a measurement
        nobody made. Same pixels in, same pixels out."""
        gapped = _with_gap()

        levelled = flatten_plane(gapped, allow_gaps=True)

        np.testing.assert_array_equal(np.isfinite(levelled), np.isfinite(gapped))

    def test_a_plane_needs_three_points(self) -> None:
        z = np.full((8, 8), np.nan)
        z[0, 0] = z[1, 1] = 1.0

        with pytest.raises(InvalidImageError, match="at least three"):
            flatten_plane(z, allow_gaps=True)


class TestTheRowsFitAroundTheGap:
    def test_a_partially_gapped_row_is_levelled_from_what_is_left(self) -> None:
        z = _scene()
        z[20, 10:20] = np.nan

        levelled = flatten_lines(z, allow_gaps=True)

        assert np.isfinite(levelled[20]).sum() == z.shape[1] - 10
        assert np.abs(np.nan_to_num(levelled[20])).max() > 0

    def test_a_fully_gapped_row_comes_back_absent(self) -> None:
        """Which is what a dropped scan line is. Absent, not zero."""
        levelled = flatten_lines(_with_gap(), allow_gaps=True)

        assert not np.isfinite(levelled[list(GAP_ROWS)]).any()
        assert np.isfinite(levelled[[0, 29, 32, 63]]).all()

    def test_the_unfittable_rows_are_counted_out_loud(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A scan that lost half its lines must not level silently — rows
        vanishing without a reason is how B-059 stayed invisible."""
        with caplog.at_level(logging.WARNING):
            flatten_lines(_with_gap(), allow_gaps=True)

        assert "2 of 64 rows" in caplog.text
        assert "lost feedback" in caplog.text

    def test_an_intact_scan_says_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            flatten_lines(_scene(), allow_gaps=True)

        assert caplog.text == ""

    def test_a_row_too_sparse_for_the_order_is_a_gap_not_an_error(self) -> None:
        """A row with one finite point cannot carry a linear fit. That is a gap
        in the data, not a malformed request, so it is absent rather than
        raising — unlike a row that is genuinely too *short*, which is
        `InvalidParameterError` and unchanged."""
        z = _scene()
        z[5, 1:] = np.nan

        levelled = flatten_lines(z, allow_gaps=True)

        assert not np.isfinite(levelled[5]).any()


class TestWhatThisDoesNotDo:
    def test_the_levelled_output_is_still_refused_downstream(self) -> None:
        """Stated as a test so the limitation cannot be forgotten: the result
        carries NaN, so the substrate step still refuses it. Nothing here has
        decided what a substrate under a gap means — that is B-065."""
        from nanoscope.core.science.preprocessing import build_substrate_map

        levelled = flatten_lines(flatten_plane(_with_gap(), allow_gaps=True), allow_gaps=True)

        with pytest.raises(InvalidImageError, match="not finite"):
            build_substrate_map(levelled, pixel_size_nm=2.0)
