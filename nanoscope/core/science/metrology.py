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

from nanoscope.core.errors import InvalidParameterError


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
