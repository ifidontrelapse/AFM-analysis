"""Laplacian of Gaussian (LoG) detector and helper functions.

**Shim.** The implementations moved to `nanoscope.core.science.detection.log` in
M2-T05. Re-exports only; the characterization harness imports four of these names
directly. Deleted in M2-T15.
"""

from __future__ import annotations

from nanoscope.core.science.detection.log import (
    LogDetector,
    detect_particles,
    estimate_log_params,
    estimate_log_threshold,
    estimate_log_threshold_adaptive,
)

__all__ = [
    "LogDetector",
    "detect_particles",
    "estimate_log_params",
    "estimate_log_threshold",
    "estimate_log_threshold_adaptive",
]
