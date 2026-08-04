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
    radius_nm: float
    confidence: float = 1.0
    bbox: tuple[int, int, int, int] = field(default_factory=tuple)  # x1,y1,x2,y2
