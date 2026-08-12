"""Reading images off a disk — the part of the old `afm_io` that touches the world.

Moved verbatim from `src/afm_io.py` in M2-T04. It lives in `infrastructure`
because every function here takes a path and opens it; `cv2` and `np.load` are
adapters, not domain. The SPM decoding it calls stays in
`nanoscope.core.science.io`.

`make_synthetic_afm` came across in M2-T04 as a `pass` stub and was deleted in
M2-T13: nothing imported it and it had no body.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from nanoscope.core.entities import AFMRawData, MicroscopyData
from nanoscope.core.errors import DataFormatError, InvalidParameterError, MissingFileError
from nanoscope.core.science.io.nanoscope_spm import _read_nanoscope_z


def _given_and_positive(value: float | None, name: str) -> float | None:
    """Accept a physical size the caller supplied, or `None` for "unknown".

    `None` is a state, not a missing value (ADR-0019), so it passes through. A
    number that *is* given has to be a real size: zero and negative are caller
    errors rather than another spelling of "unknown", and the `or` this replaces
    turned both into a fabricated default (ADR-0025). Restates the rule in
    `PixelScale.__post_init__`, at the boundary where the value enters.

    Args:
        value: the caller's number, or None
        name:  parameter name, for the error message

    Returns:
        The value unchanged, or None.

    Raises:
        ValueError: if a value is given and is not strictly positive. `nan`
            fails the same comparison, which is the intent.
    """
    if value is None:
        return None
    if not value > 0:
        raise InvalidParameterError(f"{name} must be positive when given, got {value!r}")
    return value


def load_afm(
    file_path: str,
    fmt: str,
    pixel_size_nm: float | None = None,
    scan_size_nm: float | None = None,
) -> AFMRawData:
    """
    Load a raw AFM file.

    For "spm": pixel_size_nm and scan_size_nm are extracted from the file header.
    For "npy": no metadata is stored in the file — pass them explicitly, or
               leave them out and the scale is **unknown**, which is `None`
               through to the result. Nothing is invented (ADR-0025).

    Args:
        file_path:     path to the file
        fmt:           "spm" or "npy"
        pixel_size_nm: nm/pixel for "npy"; None means unknown. Ignored for "spm"
        scan_size_nm:  full scan size in nm for "npy"; None means unknown.
                       Ignored for "spm"

    Returns:
        AFMRawData with z_raw (float32, nm), pixel_size_nm, scan_size_nm — the
        last two `None` when the scale is unknown.

    Raises:
        ValueError: if `fmt` is not "spm" or "npy", or if a scale is given and
            is not positive.
    """
    if fmt == "spm":
        scan_size, px_size, z = _read_nanoscope_z(file_path)
        return AFMRawData(z_raw=z, pixel_size_nm=px_size, scan_size_nm=scan_size)

    if fmt == "npy":
        # numpy raises its own `FileNotFoundError` here, which PROJECT_RULES §3
        # forbids as a public contract — *"never let a NumPy/SciPy internal
        # error escape"* — and which no caller catching `NanoscopeError` sees.
        # Found in M5-T05 by the viewer, whose whole error handling is that
        # distinction (ADR-0030).
        try:
            z = np.load(file_path).astype(np.float32)
        except OSError as missing:
            raise MissingFileError(f"no AFM file at {file_path}: {missing}") from missing
        return AFMRawData(
            z_raw=z,
            pixel_size_nm=_given_and_positive(pixel_size_nm, "pixel_size_nm"),
            scan_size_nm=_given_and_positive(scan_size_nm, "scan_size_nm"),
        )

    raise DataFormatError(f"Unsupported format: {fmt}")


def load_microscopy_image(
    file_path: str,
    modality: Literal["sem", "tem"],
    nm_per_pixel: float | None = None,
) -> MicroscopyData:
    """
    Load a SEM or TEM image from disk. No preprocessing applied.

    Args:
        file_path:    path to image file (JPEG, PNG, TIFF, etc.)
        modality:     "sem" or "tem"
        nm_per_pixel: physical scale; None if unknown

    Returns:
        MicroscopyData
    """
    import cv2

    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise MissingFileError(f"Could not read image: {file_path}")

    return MicroscopyData(
        image=image,
        nm_per_pixel=nm_per_pixel,
        modality=modality,
    )
