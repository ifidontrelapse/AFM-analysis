"""Heights under a line, against what the notebook actually did (M7-T06, ADR-0075).

M7's third exit criterion says *"a height profile along a drawn line matches the
notebook implementation"*, and the notebook does exactly one thing:
`z_flat[y_i, x_i-half : x_i+half]` — a horizontal row slice, no interpolation.

So the criterion covers one case, and this file asserts **equality** for it. The
arbitrary line is an extension with a rule of its own, and the rest of the file
is about that rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core.errors import InvalidInputError, InvalidParameterError
from nanoscope.core.science.metrology import height_profile


def ramp(size: int = 16) -> np.ndarray:
    """A map where every pixel is distinguishable, so a wrong sample is visible."""
    return np.arange(size * size, dtype=float).reshape(size, size)


class TestItMatchesTheNotebook:
    def test_a_horizontal_line_is_the_row_slice(self) -> None:
        """`z[y, x1:x2]`, and equality rather than approximation — which is what
        one sample per pixel of length buys."""
        z = ramp()
        half = 5
        y, x = 8, 8

        _distance, heights = height_profile(z, (x - half, y), (x + half - 1, y))

        assert np.array_equal(heights, z[y, x - half : x + half])

    def test_a_vertical_line_is_the_column_slice(self) -> None:
        z = ramp()

        _distance, heights = height_profile(z, (4, 2), (4, 9))

        assert np.array_equal(heights, z[2:10, 4])

    def test_the_distance_axis_starts_at_zero_and_ends_at_the_length(self) -> None:
        distance, heights = height_profile(ramp(), (2, 3), (10, 3))

        assert distance[0] == 0.0
        assert distance[-1] == pytest.approx(8.0)
        assert distance.size == heights.size == 9


class TestTheExtension:
    def test_a_line_between_two_rows_reads_between_them(self) -> None:
        """The assertion nearest-neighbour cannot pass: half-way between a row
        of 0 and a row of 10 is 5, and a stepped profile would say 0 or 10.
        A profile made of steps is a picture of the sampling, not of the
        sample."""
        z = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])

        _distance, heights = height_profile(z, (0.0, 0.5), (2.0, 0.5))

        assert np.allclose(heights, 5.0)

    def test_a_sample_between_two_pixels_is_their_average(self) -> None:
        z = np.array([[0.0, 10.0]])

        _distance, heights = height_profile(z, (0.0, 0.0), (1.0, 0.0))

        assert heights[0] == 0.0
        assert heights[-1] == 10.0

    def test_it_clamps_rather_than_extrapolating(self) -> None:
        """A line whose end is outside the scan is an operator's aim; inventing
        values beyond the data would be a measurement of nothing."""
        z = ramp(4)

        _distance, heights = height_profile(z, (0.0, 0.0), (99.0, 0.0))

        assert np.isfinite(heights).all()
        assert heights.max() <= z.max()


class TestWhatItRefuses:
    def test_a_line_of_no_length_is_not_a_profile(self) -> None:
        with pytest.raises(InvalidParameterError, match="two different points"):
            height_profile(ramp(), (3.0, 3.0), (3.0, 3.0))

    def test_it_validates_its_map_like_every_other_entry_point(self) -> None:
        """ADR-0030's funnel, at its fifteenth site: a profile of a 3-D array or
        of a NaN map is a wrong answer waiting to be plotted."""
        with pytest.raises(InvalidInputError):
            height_profile(np.zeros((2, 2, 2)), (0.0, 0.0), (1.0, 1.0))

        with pytest.raises(InvalidInputError):
            height_profile(np.full((4, 4), np.nan), (0.0, 0.0), (2.0, 2.0))
