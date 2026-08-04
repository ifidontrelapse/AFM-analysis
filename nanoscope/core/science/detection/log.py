"""Laplacian of Gaussian detector — pure NumPy and scikit-image, no model weights.

Moved verbatim from `src/detection/log_detector.py` in M2-T05. `detect_particles`
is the most heavily golden-covered function in the project: every phantom records
its output, so this move is checked to `rtol=1e-6` on eight images.

The YOLO detector is deliberately still in `src/`. It imports torch, which makes
it an adapter rather than domain — M2-T07 moves it to `infrastructure/models/`.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.feature import blob_log
from skimage.filters import threshold_otsu

from nanoscope.core.entities import Detection
from nanoscope.core.science.detection.base import BaseDetector

# Module-level logger, the stdlib way (M2-T11). No handler is configured here:
# a library that configures logging steals the decision from the application.
logger = logging.getLogger(__name__)


def estimate_log_params(sizes: dict) -> dict:
    """
    Derive the LoG sigma range from the output of estimate_radius_otsu.

    Radius and sigma are related by radius_px = sigma * sqrt(2). The range is
    deliberately generous: LoG finds the best sigma per particle within it.

    Args:
        sizes: dict from estimate_radius_otsu (contains radii_px)

    Returns:
        dict with min_sigma and max_sigma, in pixels
    """
    radii_px = sizes["radii_px"]

    min_sigma = radii_px.min() / np.sqrt(2) * 0.5  # half the smallest radius
    max_sigma = radii_px.max() / np.sqrt(2) * 2.0  # twice the largest radius

    # Guard against degenerate cases
    min_sigma = max(min_sigma, 1.0)
    max_sigma = max(max_sigma, min_sigma * 2)

    return {
        "min_sigma": min_sigma,
        "max_sigma": max_sigma,
    }


def estimate_log_threshold(z_above: np.ndarray) -> float:
    """
    Automatic LoG threshold, derived from the substrate noise.

    Pixels below the Otsu threshold are treated as substrate. The threshold is
    3 * the substrate noise std, normalised by the maximum of z_above.

    Args:
        z_above: z_flat - substrate (particles above the substrate)

    Returns:
        threshold for blob_log (dimensionless, 0..1)
    """
    otsu_thresh = threshold_otsu(z_above)
    substrate_px = z_above[z_above < otsu_thresh]
    noise_std = float(substrate_px.std())
    z_max = float(z_above.max())

    threshold = 3.0 * noise_std / z_max if z_max > 0 else 0.05

    return threshold


def estimate_log_threshold_adaptive(
    z_above: np.ndarray, params: dict, percentile: float = 20.0
) -> float:
    """
    Adaptive threshold, taken from the distribution of LoG responses.

    Run LoG with a minimal threshold, look at the peak response of every blob it
    finds, and take a percentile of that distribution.

    Stable across images, because it depends on the relative distribution of
    responses rather than on an absolute value.

    Args:
        z_above:    z_flat - substrate
        params:     dict from estimate_log_params (min_sigma, max_sigma)
        percentile: lower percentile of responses to cut off as noise;
                    20 discards the weakest 20%

    Returns:
        adaptive threshold for blob_log
    """
    z_norm = z_above / z_above.max()

    # Find every blob at a minimal threshold
    raw = blob_log(
        z_norm,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        num_sigma=15,
        threshold=0.01,
        overlap=0.9,
    )

    if len(raw) == 0:
        return 0.05

    # Peak response in the neighbourhood of each blob centre
    responses = []
    for blob in raw:
        y, x, sigma = blob
        r = max(int(sigma), 1)
        y1 = max(0, int(y) - r)
        y2 = min(z_norm.shape[0], int(y) + r)
        x1 = max(0, int(x) - r)
        x2 = min(z_norm.shape[1], int(x) + r)
        responses.append(float(z_norm[y1:y2, x1:x2].max()))

    responses = np.array(responses)
    threshold = float(np.percentile(responses, percentile))

    logger.debug(
        "adaptive LoG threshold: %d responses in [%.3f, %.3f], p%.0f = %.4f",
        len(responses),
        responses.min(),
        responses.max(),
        percentile,
        threshold,
    )

    return threshold


def detect_particles(
    z_above: np.ndarray,
    pixel_size_nm: float,
    sizes: dict,
    overlap: float = 0.3,
    threshold: float = None,
    percentile: float = 20.0,
) -> np.ndarray:
    """
    Detect particles with a Laplacian of Gaussian filter.

    Steps:
        1. Derive the sigma range from the Otsu radii
        2. Derive the threshold from the substrate noise
        3. Normalise z_above to [0, 1] — LoG is sensitive to scale
        4. Run blob_log
        5. Attach the physical radius in nm

    Args:
        z_above:       z_flat - substrate (particles above the substrate)
        pixel_size_nm: nm per pixel
        sizes:         dict from estimate_radius_otsu
        overlap:       permitted blob overlap (0..1)

    Returns:
        blobs: np.ndarray shape (N, 4) — [y, x, sigma_px, radius_nm]
    """
    params = estimate_log_params(sizes)
    if threshold is None:
        threshold = estimate_log_threshold_adaptive(z_above, params, percentile)

    # LoG runs on an image normalised to [0, 1]
    z_norm = z_above / z_above.max()

    raw_blobs = blob_log(
        z_norm,
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        num_sigma=15,
        threshold=threshold,
        overlap=overlap,
    )

    if len(raw_blobs) == 0:
        logger.warning("no particles found; try lowering the threshold")
        return np.empty((0, 4))

    # radius = sigma * sqrt(2) — the standard LoG relation
    sigma_px = raw_blobs[:, 2]
    radius_nm = sigma_px * np.sqrt(2) * pixel_size_nm

    blobs = np.column_stack(
        [
            raw_blobs[:, :2],  # y, x
            sigma_px,  # sigma in pixels
            radius_nm,  # radius in nm
        ]
    )

    blobs = _filter_boundary_blobs(blobs, z_above.shape)

    logger.info(
        "found %d particles: radius %.1f-%.1f nm (median %.1f), "
        "LoG threshold %.4f, sigma %.1f-%.1f px",
        len(blobs),
        radius_nm.min(),
        radius_nm.max(),
        np.median(radius_nm),
        threshold,
        params["min_sigma"],
        params["max_sigma"],
    )

    return blobs


def _filter_boundary_blobs(blobs: np.ndarray, shape: tuple, margin: float = 1.0) -> np.ndarray:
    """
    Drop particles whose circle extends past the edge of the image.

    Args:
        blobs:  (N, 4) — [y, x, sigma, radius_nm]
        shape:  (height, width) of the image
        margin: extra padding in pixels (default 1)
    Returns:
        the filtered blob array
    """
    h, w = shape
    y, x, sigma = blobs[:, 0], blobs[:, 1], blobs[:, 2]
    radius_px = sigma * np.sqrt(2)

    valid = (
        (y - radius_px >= margin)  # top edge
        & (y + radius_px <= h - margin)  # bottom edge
        & (x - radius_px >= margin)  # left edge
        & (x + radius_px <= w - margin)  # right edge
    )

    return blobs[valid]


class LogDetector(BaseDetector):
    """
    Laplacian of Gaussian (LoG) particle detector.
    Wraps the existing detect_particles function.
    """

    def __init__(
        self,
        overlap: float = 0.3,
        threshold: float | None = None,
        percentile: float = 20.0,
    ):
        self.overlap = overlap
        self.threshold = threshold
        self.percentile = percentile
        self._last_blobs: np.ndarray = np.empty((0, 4))

    def detect(
        self,
        z_above: np.ndarray,
        pixel_size_nm: float,
        sizes: dict | None = None,
    ) -> list[Detection]:
        """
        Args:
            z_above:       z_flat - substrate
            pixel_size_nm: nm/pixel
            sizes:         dict from estimate_radius_otsu (needed for sigma range).
                           If None, estimated automatically via Otsu.
        """
        if sizes is None:
            from skimage.measure import label, regionprops

            thresh = threshold_otsu(z_above)
            binary = z_above > thresh
            props = regionprops(label(binary))
            radii_px = np.array([p.equivalent_diameter_area / 2 for p in props])
            sizes = {"radii_px": radii_px}

        blobs = detect_particles(
            z_above,
            pixel_size_nm,
            sizes,
            overlap=self.overlap,
            threshold=self.threshold,
            percentile=self.percentile,
        )
        self._last_blobs = blobs
        return self._blobs_to_detections(blobs)

    @property
    def last_blobs(self) -> np.ndarray:
        """Raw (N, 4) array from the last detect() call — for SAM2 and measure_all_baseline."""
        return self._last_blobs
