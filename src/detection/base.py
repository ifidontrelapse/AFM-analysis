"""Base types shared by all detectors.

**Shim.** `BaseDetector` moved to `nanoscope.core.science.detection` in M2-T05.
Re-exports only. Deleted in M2-T15.
"""

from __future__ import annotations

from nanoscope.core.entities import Detection
from nanoscope.core.science.detection import BaseDetector

__all__ = ["BaseDetector", "Detection"]
