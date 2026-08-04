"""YOLOv8 particle detector.

**Shim.** `YoloDetector` moved to `nanoscope.infrastructure.models` in M2-T07 —
it needs model weights, which makes it an adapter, not domain. Re-exports only.
Deleted in M2-T15.
"""

from __future__ import annotations

from nanoscope.infrastructure.models import YoloDetector

__all__ = ["YoloDetector"]
