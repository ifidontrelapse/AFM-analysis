"""Base types shared by all detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.types import Detection


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
            detections.append(Detection(
                x_px=float(x),
                y_px=float(y),
                radius_px=radius_px,
                radius_nm=float(radius_nm),
                bbox=(
                    int(x - radius_px), int(y - radius_px),
                    int(x + radius_px), int(y + radius_px),
                ),
            ))
        return detections
