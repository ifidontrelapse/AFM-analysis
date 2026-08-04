"""
Deterministic synthetic microscopy phantoms for characterization testing.

Every phantom is a pure function of its arguments and a seed. No file I/O, no
randomness that is not seeded, no dependency on anything under ``src/``.

Ground truth is returned alongside the image so that a future evaluation
harness can score detection/segmentation against it. Phase 0 only records the
*current* behaviour of the code; it does not assert that the current behaviour
matches ground truth.

Units follow the repository invariant: ``_nm`` for nanometres, ``_px`` for
pixels. Heights are nanometres. ``pixel_size_nm`` is nm per pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Phantom:
    """A synthetic image plus the ground truth used to build it."""

    name: str
    image: NDArray[np.float32]
    pixel_size_nm: float
    scan_size_nm: float
    #: (N, 2) array of true particle centres as [y_px, x_px]
    centres_yx_px: NDArray[np.float64]
    #: (N,) true particle radii in pixels
    radii_px: NDArray[np.float64]
    #: (N,) true particle peak heights in nm above the local substrate
    heights_nm: NDArray[np.float64]
    notes: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def radii_nm(self) -> NDArray[np.float64]:
        return self.radii_px * self.pixel_size_nm

    @property
    def n_particles(self) -> int:
        return len(self.radii_px)


def _gaussian_caps(
    shape: tuple[int, int],
    centres_yx: NDArray[np.float64],
    radii_px: NDArray[np.float64],
    heights_nm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum of Gaussian caps. sigma = radius / 1.5 puts the nominal radius near
    the visual edge of the cap (~1.5 sigma), which matches how a spherical
    particle images under an AFM tip."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    z = np.zeros(shape, dtype=np.float64)
    for (cy, cx), r, h in zip(centres_yx, radii_px, heights_nm, strict=True):
        sigma = r / 1.5
        z += h * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    return z


def _poisson_disc_centres(
    shape: tuple[int, int],
    n: int,
    min_sep_px: float,
    margin_px: float,
    rng: np.random.Generator,
    max_tries: int = 20000,
) -> NDArray[np.float64]:
    """Rejection-sampled centres with a minimum separation, so particle counts
    are reproducible and non-overlapping unless overlap is asked for."""
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        cy = rng.uniform(margin_px, shape[0] - margin_px)
        cx = rng.uniform(margin_px, shape[1] - margin_px)
        if all((cy - py) ** 2 + (cx - px) ** 2 >= min_sep_px**2 for py, px in pts):
            pts.append((cy, cx))
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


# ── AFM phantoms ──────────────────────────────────────────────────────────────


def afm_flat_monodisperse(
    size: int = 256,
    n: int = 24,
    radius_px: float = 7.0,
    height_nm: float = 18.0,
    noise_nm: float = 0.12,
    pixel_size_nm: float = 2.0,
    seed: int = 0,
) -> Phantom:
    """Ideal case: flat substrate, identical well-separated particles.

    This is the phantom against which any regression is least ambiguous — if
    numbers move here, the change is real."""
    rng = np.random.default_rng(seed)
    centres = _poisson_disc_centres(
        (size, size), n, min_sep_px=radius_px * 3.2, margin_px=radius_px * 2.5, rng=rng
    )
    radii = np.full(len(centres), radius_px)
    heights = np.full(len(centres), height_nm)
    z = _gaussian_caps((size, size), centres, radii, heights)
    z += rng.normal(0.0, noise_nm, (size, size))
    return Phantom(
        name="afm_flat_monodisperse",
        image=z.astype(np.float32),
        pixel_size_nm=pixel_size_nm,
        scan_size_nm=size * pixel_size_nm,
        centres_yx_px=centres,
        radii_px=radii,
        heights_nm=heights,
        notes="Flat substrate, monodisperse, well separated. Baseline sanity case.",
    )


def afm_tilted_polydisperse(
    size: int = 256,
    n: int = 30,
    radius_range_px: tuple[float, float] = (4.0, 12.0),
    height_range_nm: tuple[float, float] = (8.0, 30.0),
    noise_nm: float = 0.2,
    tilt_nm_per_px: tuple[float, float] = (0.05, 0.03),
    line_artefact_nm: float = 0.8,
    pixel_size_nm: float = 2.0,
    seed: int = 1,
) -> Phantom:
    """Realistic case: sample tilt, per-line offsets (scanner drift), and a
    polydisperse population. Exercises flatten_plane + flatten_lines."""
    rng = np.random.default_rng(seed)
    r_max = radius_range_px[1]
    centres = _poisson_disc_centres(
        (size, size), n, min_sep_px=r_max * 2.6, margin_px=r_max * 2.2, rng=rng
    )
    radii = rng.uniform(*radius_range_px, len(centres))
    heights = rng.uniform(*height_range_nm, len(centres))

    z = _gaussian_caps((size, size), centres, radii, heights)
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    z += tilt_nm_per_px[0] * xx + tilt_nm_per_px[1] * yy  # global plane tilt
    z += rng.normal(0.0, line_artefact_nm, (size, 1))  # per-line offset
    z += rng.normal(0.0, noise_nm, (size, size))
    return Phantom(
        name="afm_tilted_polydisperse",
        image=z.astype(np.float32),
        pixel_size_nm=pixel_size_nm,
        scan_size_nm=size * pixel_size_nm,
        centres_yx_px=centres,
        radii_px=radii,
        heights_nm=heights,
        notes="Plane tilt + per-line offsets + polydisperse. Exercises flattening.",
        meta={"tilt_nm_per_px": tilt_nm_per_px, "line_artefact_nm": line_artefact_nm},
    )


def afm_dense_overlapping(
    size: int = 256,
    n: int = 70,
    radius_px: float = 6.0,
    height_nm: float = 15.0,
    noise_nm: float = 0.15,
    pixel_size_nm: float = 2.0,
    seed: int = 2,
) -> Phantom:
    """Dense field where particles touch and overlap. Stresses LoG overlap
    handling and the ring-baseline fallback (rings get eaten by neighbours)."""
    rng = np.random.default_rng(seed)
    centres = _poisson_disc_centres(
        (size, size), n, min_sep_px=radius_px * 1.35, margin_px=radius_px * 2.0, rng=rng
    )
    radii = np.full(len(centres), radius_px)
    heights = np.full(len(centres), height_nm)
    z = _gaussian_caps((size, size), centres, radii, heights)
    z += rng.normal(0.0, noise_nm, (size, size))
    return Phantom(
        name="afm_dense_overlapping",
        image=z.astype(np.float32),
        pixel_size_nm=pixel_size_nm,
        scan_size_nm=size * pixel_size_nm,
        centres_yx_px=centres,
        radii_px=radii,
        heights_nm=heights,
        notes="Dense/overlapping. Stresses NMS-like overlap and ring baselines.",
    )


def afm_sparse_low_snr(
    size: int = 256,
    n: int = 6,
    radius_px: float = 5.0,
    height_nm: float = 3.0,
    noise_nm: float = 1.0,
    pixel_size_nm: float = 2.0,
    seed: int = 3,
) -> Phantom:
    """Low signal-to-noise: peak height is 3x the noise sigma. Detector
    thresholds are most fragile here."""
    rng = np.random.default_rng(seed)
    centres = _poisson_disc_centres(
        (size, size), n, min_sep_px=radius_px * 6.0, margin_px=radius_px * 3.0, rng=rng
    )
    radii = np.full(len(centres), radius_px)
    heights = np.full(len(centres), height_nm)
    z = _gaussian_caps((size, size), centres, radii, heights)
    z += rng.normal(0.0, noise_nm, (size, size))
    return Phantom(
        name="afm_sparse_low_snr",
        image=z.astype(np.float32),
        pixel_size_nm=pixel_size_nm,
        scan_size_nm=size * pixel_size_nm,
        centres_yx_px=centres,
        radii_px=radii,
        heights_nm=heights,
        notes="SNR ~3. Threshold selection is most fragile here.",
    )


def afm_coarse_pixels(
    size: int = 128,
    n: int = 14,
    radius_px: float = 4.0,
    height_nm: float = 20.0,
    noise_nm: float = 0.15,
    pixel_size_nm: float = 9.77,
    seed: int = 4,
) -> Phantom:
    """Coarse pixel scale, matching the median of the operator's real scans
    (~9.8 nm/px). At this scale ``int(min_size_nm / pixel_size_nm)`` floors to
    zero and the minimum-particle-size noise filter silently disengages."""
    rng = np.random.default_rng(seed)
    centres = _poisson_disc_centres(
        (size, size), n, min_sep_px=radius_px * 3.5, margin_px=radius_px * 2.5, rng=rng
    )
    radii = np.full(len(centres), radius_px)
    heights = np.full(len(centres), height_nm)
    z = _gaussian_caps((size, size), centres, radii, heights)
    z += rng.normal(0.0, noise_nm, (size, size))
    return Phantom(
        name="afm_coarse_pixels",
        image=z.astype(np.float32),
        pixel_size_nm=pixel_size_nm,
        scan_size_nm=size * pixel_size_nm,
        centres_yx_px=centres,
        radii_px=radii,
        heights_nm=heights,
        notes="pixel_size_nm > min_size_nm -> min_size_pixel floors to 0 (defect D-08).",
    )


# ── SEM / TEM phantoms ────────────────────────────────────────────────────────


def sem_bright_particles(
    size: int = 256,
    n: int = 22,
    radius_px: float = 8.0,
    pixel_size_nm: float | None = 1.5,
    seed: int = 10,
) -> Phantom:
    """SEM-like: bright particles on a dark background, 8-bit valued."""
    rng = np.random.default_rng(seed)
    centres = _poisson_disc_centres(
        (size, size), n, min_sep_px=radius_px * 3.0, margin_px=radius_px * 2.2, rng=rng
    )
    radii = np.full(len(centres), radius_px)
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    img = np.full((size, size), 40.0)
    for (cy, cx), r in zip(centres, radii, strict=True):
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r**2
        img[disc] = 210.0
    img += rng.normal(0.0, 6.0, (size, size))
    img = np.clip(img, 0, 255)
    return Phantom(
        name="sem_bright_particles",
        image=img.astype(np.float32),
        pixel_size_nm=pixel_size_nm if pixel_size_nm is not None else float("nan"),
        scan_size_nm=size * (pixel_size_nm or 1.0),
        centres_yx_px=centres,
        radii_px=radii,
        heights_nm=np.full(len(centres), np.nan),
        notes="Bright-on-dark, 8-bit range. No height channel.",
        meta={"dtype_hint": "uint8", "polarity": "bright"},
    )


def tem_dark_particles(
    size: int = 256,
    n: int = 22,
    radius_px: float = 8.0,
    pixel_size_nm: float | None = 0.5,
    seed: int = 11,
) -> Phantom:
    """TEM-like: DARK particles on a bright background. The current LoG path
    assumes particles are bright, so this phantom is the modality-polarity
    counter-example."""
    rng = np.random.default_rng(seed)
    centres = _poisson_disc_centres(
        (size, size), n, min_sep_px=radius_px * 3.0, margin_px=radius_px * 2.2, rng=rng
    )
    radii = np.full(len(centres), radius_px)
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    img = np.full((size, size), 205.0)
    for (cy, cx), r in zip(centres, radii, strict=True):
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r**2
        img[disc] = 45.0
    img += rng.normal(0.0, 6.0, (size, size))
    img = np.clip(img, 0, 255)
    return Phantom(
        name="tem_dark_particles",
        image=img.astype(np.float32),
        pixel_size_nm=pixel_size_nm if pixel_size_nm is not None else float("nan"),
        scan_size_nm=size * (pixel_size_nm or 1.0),
        centres_yx_px=centres,
        radii_px=radii,
        heights_nm=np.full(len(centres), np.nan),
        notes="Dark-on-bright. Counter-example to the bright-particle assumption.",
        meta={"dtype_hint": "uint8", "polarity": "dark"},
    )


# ── Degenerate inputs ─────────────────────────────────────────────────────────


def degenerate_inputs() -> dict[str, NDArray[np.float32]]:
    """Inputs that a hardened pipeline must reject with a typed, actionable
    error. Phase 0 records whatever the code does today."""
    rng = np.random.default_rng(99)
    return {
        "empty": np.zeros((0, 0), dtype=np.float32),
        "single_pixel": np.array([[1.0]], dtype=np.float32),
        "constant_zero": np.zeros((64, 64), dtype=np.float32),
        "constant_nonzero": np.full((64, 64), 7.5, dtype=np.float32),
        "all_negative": np.full((64, 64), -5.0, dtype=np.float32),
        "with_nan": np.where(
            np.arange(64 * 64).reshape(64, 64) == 100,
            np.nan,
            rng.normal(0, 1, (64, 64)),
        ).astype(np.float32),
        "with_inf": np.where(
            np.arange(64 * 64).reshape(64, 64) == 100,
            np.inf,
            rng.normal(0, 1, (64, 64)),
        ).astype(np.float32),
        "one_dimensional": np.arange(64, dtype=np.float32),
        "three_dimensional": np.zeros((8, 8, 3), dtype=np.float32),
        "extreme_aspect": rng.normal(0, 1, (2, 4096)).astype(np.float32),
    }


ALL_AFM_PHANTOMS = (
    afm_flat_monodisperse,
    afm_tilted_polydisperse,
    afm_dense_overlapping,
    afm_sparse_low_snr,
    afm_coarse_pixels,
)

ALL_IMAGE_PHANTOMS = (
    sem_bright_particles,
    tem_dark_particles,
)
