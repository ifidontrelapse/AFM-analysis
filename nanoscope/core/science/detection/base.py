"""What every detector answers with, and the blob-to-Detection conversion.

Moved verbatim from `src/detection/base.py` in M2-T05.

`BaseDetector` is an ABC inside the science layer, not the `Detector` port —
M2-T08 defines that, and it is a different thing: the port is what `application`
depends on, this is shared implementation between the LoG and YOLO detectors.
`_blobs_to_detections` carries the `radius_px = sigma * sqrt(2)` relation, which
is physics and is recorded in the golden.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from nanoscope.core.entities import Detection


class BaseDetector(ABC):
    """Abstract base class for all particle detectors."""

    @abstractmethod
    def detect(self, z_above: np.ndarray, pixel_size_nm: float) -> list[Detection]:
        """
        Find particles in the image.

        Args:
            z_above:       z_flat - substrate (particles above the substrate)
            pixel_size_nm: nm/pixel
        Returns:
            list of Detection
        """
        ...

    @staticmethod
    def _blobs_to_detections(blobs: np.ndarray) -> list[Detection]:
        detections = []
        for blob in blobs:
            y, x, sigma, radius_nm = blob
            radius_px = sigma * np.sqrt(2)
            detections.append(
                Detection(
                    x_px=float(x),
                    y_px=float(y),
                    radius_px=radius_px,
                    radius_nm=float(radius_nm),
                    bbox=(
                        int(x - radius_px),
                        int(y - radius_px),
                        int(x + radius_px),
                        int(y + radius_px),
                    ),
                )
            )
        return detections
