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
from nanoscope.core.errors import InvalidParameterError
from nanoscope.core.science.detection.base import BaseDetector
from nanoscope.core.validation import ensure_height_map, ensure_positive
from nanoscope.core.values import Polarity

# Module-level logger, the stdlib way (M2-T11). No handler is configured here:
# a library that configures logging steals the decision from the application.
logger = logging.getLogger(__name__)

#: Threshold used when one cannot be derived from the image — a flat map, or a
#: map with no positive signal above the substrate. The value is the one this
#: module has always used in that situation; ADR-0018 only gave it a name and a
#: third call site.
DEFAULT_THRESHOLD = 0.05


def _ensure_percentile(percentile: float) -> None:
    """`np.percentile` accepts only [0, 100], and answers anything else with a
    message about `q`, a name no caller of this module ever typed."""
    if not 0.0 <= percentile <= 100.0:
        raise InvalidParameterError(f"percentile must be between 0 and 100, got {percentile!r}.")


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
    if not isinstance(sizes, dict) or "radii_px" not in sizes:
        raise InvalidParameterError(
            "sizes must be the dict estimate_radius_otsu returns, with a 'radii_px' entry; "
            f"got {type(sizes).__name__}"
            + (f" with keys {sorted(sizes)}" if isinstance(sizes, dict) else "")
            + "."
        )
    radii_px = np.asarray(sizes["radii_px"])
    if radii_px.size == 0:
        raise InvalidParameterError(
            "sizes['radii_px'] is empty, so there is no sigma range to derive. "
            "estimate_radius_otsu raises rather than returning an empty set (ADR-0017); "
            "a hand-built sizes dict has to hold at least one radius."
        )

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
    z_above = ensure_height_map(z_above, "z_above")

    otsu_thresh = threshold_otsu(z_above)
    substrate_px = z_above[z_above < otsu_thresh]
    noise_std = float(substrate_px.std())
    z_max = float(z_above.max())

    threshold = 3.0 * noise_std / z_max if z_max > 0 else DEFAULT_THRESHOLD

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

    A map with no positive signal cannot be normalised into [0, 1], so the
    conservative default is returned instead of a threshold derived from a
    division by zero (ADR-0018 / D-11).
    """
    z_above = ensure_height_map(z_above, "z_above")
    _ensure_percentile(percentile)

    z_max = float(z_above.max())
    if not z_max > 0:
        logger.warning(
            "no positive signal above the substrate (max = %.3g); using the default threshold %.2f",
            z_max,
            DEFAULT_THRESHOLD,
        )
        return DEFAULT_THRESHOLD

    z_norm = z_above / z_max

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
        return DEFAULT_THRESHOLD

    # Peak response in the neighbourhood of each blob centre
    peaks: list[float] = []
    for blob in raw:
        y, x, sigma = blob
        r = max(int(sigma), 1)
        y1 = max(0, int(y) - r)
        y2 = min(z_norm.shape[0], int(y) + r)
        x1 = max(0, int(x) - r)
        x2 = min(z_norm.shape[1], int(x) + r)
        peaks.append(float(z_norm[y1:y2, x1:x2].max()))

    responses = np.array(peaks)
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
    pixel_size_nm: float | None,
    sizes: dict,
    overlap: float = 0.3,
    threshold: float | None = None,
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
        pixel_size_nm: nm per pixel, or None when the physical scale is unknown
        sizes:         dict from estimate_radius_otsu
        overlap:       permitted blob overlap (0..1)

    Returns:
        blobs: np.ndarray shape (N, 4) — [y, x, sigma_px, radius_nm].
        `radius_nm` is NaN throughout when `pixel_size_nm` is None.
    """
    z_above = ensure_height_map(z_above, "z_above")
    ensure_positive(pixel_size_nm, "pixel_size_nm", allow_none=True)
    ensure_positive(threshold, "threshold", allow_none=True)
    _ensure_percentile(percentile)
    if not 0.0 <= overlap <= 1.0:
        raise InvalidParameterError(f"overlap must be between 0 and 1, got {overlap!r}.")

    params = estimate_log_params(sizes)

    # The sizes dict is validated first, above: a caller passing nonsense should
    # hear about that rather than about the image.
    z_max = float(z_above.max())
    if not z_max > 0:
        # No positive signal above the substrate, so no particles above it —
        # zero is the honest answer, not a NaN image that blob_log silently
        # finds nothing in (ADR-0018 / D-11). `not z_max > 0` also catches a NaN
        # maximum, which the old division propagated into every pixel.
        logger.warning(
            "no positive signal above the substrate (max = %.3g); no particles to find", z_max
        )
        return np.empty((0, 4))

    if threshold is None:
        threshold = estimate_log_threshold_adaptive(z_above, params, percentile)

    # LoG runs on an image normalised to [0, 1]
    z_norm = z_above / z_max

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
    if pixel_size_nm is None:
        # Unknown scale: the column stays NaN rather than becoming a pixel count
        # in nanometre clothing (D-07, ADR-0019). This is the one NaN in the
        # module that is *meant* — it marks a missing measurement and never
        # enters a decision; the ones ADR-0018 removed came out of arithmetic.
        radius_nm = np.full(len(sigma_px), np.nan)
    else:
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
        "found %d particles: radius %s, LoG threshold %.4f, sigma %.1f-%.1f px",
        len(blobs),
        "unknown (no pixel scale)"
        if pixel_size_nm is None
        else f"{radius_nm.min():.1f}-{radius_nm.max():.1f} nm (median {np.median(radius_nm):.1f})",
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
        polarity: Polarity = Polarity.BRIGHT_ON_DARK,
    ):
        self.overlap = overlap
        self.threshold = threshold
        self.percentile = percentile
        # Bright-on-dark is what this detector has always assumed; naming the
        # assumption is what lets TEM work at all (D-12, ADR-0023).
        self.polarity = polarity
        self._last_blobs: np.ndarray = np.empty((0, 4))

    def detect(
        self,
        z_above: np.ndarray,
        pixel_size_nm: float | None,
        sizes: dict | None = None,
    ) -> list[Detection]:
        """
        Args:
            z_above:       z_flat - substrate
            pixel_size_nm: nm/pixel, or None when the scale is unknown — then
                           every `radius_nm` comes back None (D-07, ADR-0019)
            sizes:         dict from estimate_radius_otsu (needed for sigma range).
                           If None, estimated automatically via Otsu.

        A `DARK_ON_BRIGHT` detector inverts the image first; `sizes`, if given,
        is assumed to describe the particles either way, since a radius does not
        change sign.
        """
        # Before the inversion, not after: `z_above.max() - z_above` on a 3-D
        # array is a 3-D array, and the first thing to complain would then be
        # `blob_log`, about a shape the caller never mentioned (ADR-0030).
        z_above = ensure_height_map(z_above, "z_above")

        if self.polarity is Polarity.DARK_ON_BRIGHT:
            # One inversion, at the entrance, so everything downstream keeps the
            # single convention it was written for: particles are the high side.
            # `max - z` rather than `-z`, because the LoG path normalises by the
            # maximum and needs it positive (ADR-0018).
            z_above = z_above.max() - z_above

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
