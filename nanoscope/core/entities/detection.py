"""A single particle found in an image, in pixels and in nanometres.

Moved verbatim from `src/types.py` in M2-T02.

`bbox`'s `default_factory=tuple` produces an **empty** tuple, not the four values
the annotation promises. That is audit defect **D-16**, fixed in M3; the golden
records it as `default_detection_bbox_len: 0`, so changing it here would read as
drift from a task that is supposed to move nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    # `type: ignore` and not a fix: `default_factory=tuple` genuinely disagrees
    # with the annotation, and mypy saying so *is* D-16. Correcting it changes
    # `default_detection_bbox_len`, which the golden records, so it belongs to M3
    # with a declared numerical delta — not to a move that must shift nothing.
    # `warn_unused_ignores = true`, so this line becomes an error the moment M3
    # fixes the defect. It expires itself.
    bbox: tuple[int, int, int, int] = field(default_factory=tuple)  # type: ignore[assignment]  # D-16
