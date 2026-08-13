"""Distances an operator measured by hand (M7-T05, ADR-0074).

**A new output.** Everything else in `core.science` computes something from a
scan; this computes something from two points a person chose, and the roadmap's
risk line for M7 says what follows: *"manual measurements are a new output and
get their own tests"*.

It is here rather than in a panel because two points and Pythagoras is still
arithmetic, and arithmetic in a widget is the first science in `gui/` in seven
milestones (PROJECT_RULES §2.3).
"""

from __future__ import annotations

import math

import numpy as np

from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.validation import ensure_height_map


def distance_px(start: tuple[float, float], end: tuple[float, float]) -> float:
    """The distance between two points, in pixels.

    Zero is a real answer — an operator who clicked twice in the same place
    measured nothing, and saying so is more useful than refusing.
    """
    return math.hypot(end[0] - start[0], end[1] - start[1])


def distance_nm(
    start: tuple[float, float],
    end: tuple[float, float],
    pixel_size_nm: float | None,
) -> float | None:
    """The same distance in nanometres, or `None` when the scale is unknown.

    `None` rather than a number: a length in nanometres computed from a scale
    nobody recorded is a fabricated measurement, and this is the first surface
    in the project that *produces* a physical number rather than reading one
    (ADR-0025, ADR-0074).

    Raises:
        InvalidParameterError: a scale that is not positive. Absent is a state;
            zero or negative is a wrong answer (ADR-0025's own distinction).
    """
    if pixel_size_nm is None:
        return None
    if not pixel_size_nm > 0:
        raise InvalidParameterError(
            f"pixel_size_nm must be positive; got {pixel_size_nm!r}. An unknown scale is None"
        )
    return distance_px(start, end) * pixel_size_nm


def height_profile(
    z: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """The heights under a drawn line, sampled one per pixel of its length.

    **What the notebook did, and what this extends.**
    `notebooks/afm_gold_nanoparticles.ipynb` §5.1 takes `z[y, x1:x2]` — a
    horizontal row slice, no interpolation. That is the case M7's exit criterion
    names, and for a horizontal line with integer endpoints this returns exactly
    those pixels, which is why the equality can be asserted rather than
    approximated.

    An arbitrary line has no notebook counterpart, so it is an extension with a
    stated rule: **bilinear**, because a diagonal profile made of
    nearest-neighbour steps is a picture of the sampling rather than of the
    sample (ADR-0075).

    Args:
        z: the height map, indexed `[y, x]` like every array in this project.
        start: `(x, y)` in pixels — the order a screen speaks in.
        end: the other end.

    Returns:
        `(distance_px, heights)`, both one-dimensional and the same length.
        `distance_px` starts at zero.

    Raises:
        InvalidInputError: `z` is not a height map this library can compute on
            (ADR-0030's funnel, at its fifteenth entry point).
        InvalidParameterError: both ends are the same point — a profile of no
            length is not a profile, which is the refusal a zero-length ruler
            already gets.
    """
    z = ensure_height_map(z, "z")
    length = distance_px(start, end)
    if not length > 0:
        raise InvalidParameterError(f"a profile needs two different points; both ends are {start}")

    #: One sample per pixel of length, **plus the far end**: a horizontal line
    #: from x=10 to x=14 crosses five pixels, not four.
    samples = int(round(length)) + 1
    xs = np.linspace(start[0], end[0], samples)
    ys = np.linspace(start[1], end[1], samples)
    return np.linspace(0.0, length, samples), _bilinear(z, xs, ys)


def _bilinear(z: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """`z` sampled at `(x, y)`, bilinearly, clamped to the array's edges.

    Clamped rather than extrapolated: a line whose end is half a pixel outside
    the scan is an operator's aim, and inventing values beyond the data would be
    a measurement of nothing.
    """
    height, width = z.shape
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = xs - x0
    fy = ys - y0

    top = z[y0, x0] * (1 - fx) + z[y0, x1] * fx
    bottom = z[y1, x0] * (1 - fx) + z[y1, x1] * fx
    return np.asarray(top * (1 - fy) + bottom * fy, dtype=float)
