"""What a hand-drawn line measures (M7-T05, ADR-0074).

The arithmetic is `core.science.metrology`; what is here is the pairing of a
ruler with **the scale the project recorded for its image**, which is the same
join `run_analysis` makes and the same one M4-T05 found a defect hiding in.

It exists at all because `gui/` may not import `core.science` (Architecture §3.2,
checked since M5-T06) — the third time that guard has sent a number through this
layer instead of around it.
"""

from __future__ import annotations

from nanoscope.core.entities import Ruler
from nanoscope.core.science.metrology import distance_nm, distance_px


def ruler_length(ruler: Ruler, pixel_size_nm: float | None) -> tuple[float, float | None]:
    """`(pixels, nanometres)` for one line — the second `None` without a scale.

    Computed from the endpoints every time, never read from a row: a stored
    length is a second answer waiting to disagree with the points it came from.
    """
    return (
        distance_px(ruler.start, ruler.end),
        distance_nm(ruler.start, ruler.end, pixel_size_nm),
    )
