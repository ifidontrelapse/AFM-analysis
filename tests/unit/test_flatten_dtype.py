"""Levelling returns the residuals it computed, whatever dtype it was given
(D-13, ADR-0029).

`flatten_lines` pre-allocated with `np.empty_like(z)`, so the float64 residuals
`np.polyfit`/`np.polyval` produce were cast back into the input's dtype on
assignment. An integer image levelled to **all zeros** — every residual of a
row's own linear fit is fractional — and a boolean one came back as a *mask* of
where the residual was non-zero.

`flatten_plane` never had the defect: it returns `z - plane` and lets NumPy
promote. These tests hold the two functions to the same rule, which is the one
the task was named for.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core.science.preprocessing import flatten_lines, flatten_plane

DTYPES = ["uint8", "int16", "int32", "bool", "float32", "float64"]


def _tilted_ramp(h: int = 32, w: int = 32) -> np.ndarray:
    """A map with a per-row trend to remove and structure to keep. Scaled to
    0..255 so it survives a `uint8` cast, which is what an SEM/TEM image is."""
    ys, xs = np.mgrid[0:h, 0:w]
    z = 3.0 * xs + 1.5 * ys
    z += 40.0 * np.exp(-((ys - h // 2) ** 2 + (xs - w // 2) ** 2) / (2 * 3.0**2))
    return 255.0 * (z - z.min()) / (z.max() - z.min())


def _reference(z: np.ndarray, poly_order: int = 1) -> np.ndarray:
    """The residuals, computed the way the function computes them but never
    stored anywhere narrower than float64."""
    xi = np.arange(z.shape[1])
    return np.stack([row - np.polyval(np.polyfit(xi, row, poly_order), xi) for row in z])


def test_an_integer_image_keeps_its_residuals() -> None:
    """The defect as the audit measured it: an integer output rounded every
    residual toward zero, so levelling returned nothing at all."""
    z = _tilted_ramp().astype(np.uint8)

    levelled = flatten_lines(z)

    assert np.issubdtype(levelled.dtype, np.floating)
    assert np.abs(levelled).max() > 0.5
    np.testing.assert_array_equal(levelled, _reference(z))


def test_a_boolean_image_is_not_returned_as_a_mask() -> None:
    """Worse than truncation, and unmeasured by the audit: `result[i] = <float>`
    into a bool array stores `!= 0`, so every non-zero residual became 1.0."""
    z = _tilted_ramp() > 128

    levelled = flatten_lines(z)

    assert np.issubdtype(levelled.dtype, np.floating)
    assert set(np.unique(levelled)) - {0.0, 1.0}
    np.testing.assert_array_equal(levelled, _reference(z))


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_two_halves_of_flattening_agree_about_dtype(dtype: str) -> None:
    """The invariant D-13 is a violation of. `flatten_plane` promotes because
    NumPy promotes for it; `flatten_lines` has to say so out loud."""
    z = _tilted_ramp().astype(dtype)

    assert flatten_lines(z).dtype == flatten_plane(z).dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_input_dtype_produces_the_residuals_it_computed(dtype: str) -> None:
    z = _tilted_ramp().astype(dtype)

    np.testing.assert_array_equal(flatten_lines(z), _reference(z))


def test_a_float64_map_is_untouched_by_the_fix() -> None:
    """The path every phantom takes — `flatten_plane` hands float64 on — must be
    bit-identical, which is why the golden's five AFM chains do not move."""
    z = _tilted_ramp()

    np.testing.assert_array_equal(flatten_lines(z), _reference(z))
    assert flatten_lines(z).dtype == np.float64


def test_a_float32_map_is_promoted_rather_than_rounded_to_float32() -> None:
    """The declared drift. The residuals were computed in float64 and then
    stored in float32; the difference is small, real, and one-way."""
    z = _tilted_ramp().astype(np.float32)

    levelled = flatten_lines(z)

    assert levelled.dtype == np.float64
    np.testing.assert_array_equal(levelled, _reference(z))
    assert np.abs(levelled - levelled.astype(np.float32)).max() > 0


def test_levelling_an_eight_bit_image_removes_the_row_trend() -> None:
    """What the fix buys the SEM/TEM path: `load_microscopy_image` returns
    `uint8` from `cv2.imread`, and levelling one of those images used to be a
    no-op that looked like a success."""
    z = _tilted_ramp().astype(np.uint8)
    xi = np.arange(z.shape[1])

    levelled = flatten_lines(z)
    slopes = [np.polyfit(xi, row, 1)[0] for row in levelled]

    assert np.abs(slopes).max() < 1e-12
