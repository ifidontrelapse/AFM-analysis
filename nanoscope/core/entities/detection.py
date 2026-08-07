"""A single particle found in an image, in pixels and in nanometres.

Moved verbatim from `src/types.py` in M2-T02.

`bbox` is `None` when the detector produced no box, which is the LoG path. It
used to be `field(default_factory=tuple)` — an **empty** tuple where the
annotation promised four ints (audit **D-16**), a four-element promise broken
silently. M3-T14 (ADR-0031) made it absent instead, the sixth substitute value
this milestone has deleted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """Single particle detection result."""

    x_px: float
    y_px: float
    radius_px: float
    # `None` when the image has no known pixel scale — the invariant D-07 states:
    # a physical value is either physical or absent, never a pixel count wearing
    # nanometre units (M3-T11, ADR-0019). Pixel-space fields are always present.
    radius_nm: float | None
    # `None` when the detector produces no score, which is the LoG path: its blob
    # response is not a probability and normalising one into a confidence would be
    # a scientific claim, not a fix. The old default was `1.0`, so every detection
    # — including a YOLO box that only just cleared the threshold — reported
    # certainty (audit D-09, M3-T05, ADR-0028). Absent, never a substitute value.
    confidence: float | None = None
    # `None`, not `()`: a LoG detection has no bounding box, and an empty tuple
    # is a `tuple[int, int, int, int]` that is not one — the annotation was a
    # promise the default broke (D-16, ADR-0031). The `type: ignore` that used to
    # sit here was written in M2-T02 to expire itself; `warn_unused_ignores`
    # collected it the moment this line stopped needing it.
    bbox: tuple[int, int, int, int] | None = None
