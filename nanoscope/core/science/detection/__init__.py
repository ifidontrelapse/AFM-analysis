"""Finding particles with pure NumPy — no model weights, no torch.

The LoG detector and the ABC it shares with YOLO. `YoloDetector` is not here on
purpose: it imports torch, which makes it an adapter, and M2-T07 moves it to
`nanoscope.infrastructure.models`.
"""

from nanoscope.core.science.detection.base import BaseDetector
from nanoscope.core.science.detection.log import (
    LogDetector,
    detect_particles,
    estimate_log_params,
    estimate_log_threshold,
    estimate_log_threshold_adaptive,
)

__all__ = [
    "BaseDetector",
    "LogDetector",
    "detect_particles",
    "estimate_log_params",
    "estimate_log_threshold",
    "estimate_log_threshold_adaptive",
]
