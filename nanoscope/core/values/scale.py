"""The physical size of a pixel, and the two conversions the code keeps rewriting.

`radius_px * nm_per_pixel` and `area_px * nm_per_pixel ** 2` appear by hand in
`measure.py`, `preprocess.py`, `yolo_detector.py`, `segmentation.py` and
`visualization.py`. This is that arithmetic in one place, with the one guard those
call sites do not have.

Not adopted yet — the call sites move in M2-T03…T07, each proving zero drift.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelScale:
    """Nanometres per pixel. Immutable, compared by value.

    An unknown scale is `None` at the call site, not a special instance here:
    `MicroscopyData.nm_per_pixel` and `PipelineResult.pixel_size_nm` are already
    `float | None`, and a second way to say "unknown" is a second thing to check.
    """

    nm_per_px: float

    def __post_init__(self) -> None:
        # The value comes from a parsed instrument header (`Scan Size` / samples
        # per line). A zero or negative scale silently produces zero-sized
        # particles rather than an error, which is exactly how D-01 hid.
        if not self.nm_per_px > 0:
            raise ValueError(f"pixel scale must be positive, got {self.nm_per_px!r} nm/px")

    def to_nm(self, length_px: float) -> float:
        """Convert a length in pixels to nanometres."""
        return length_px * self.nm_per_px

    def area_to_nm2(self, area_px: float) -> float:
        """Convert an area in pixels to square nanometres."""
        return area_px * self.nm_per_px**2
