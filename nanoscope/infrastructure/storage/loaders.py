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
from nanoscope.core.science.io.nanoscope_spm import _read_nanoscope_z


def load_afm(
    file_path: str,
    fmt: str,
    pixel_size_nm: float | None = None,
    scan_size_nm: float | None = None,
) -> AFMRawData:
    """
    Load a raw AFM file.

    For "spm": pixel_size_nm and scan_size_nm are extracted from the file header.
    For "npy": no metadata is stored in the file — pass them explicitly,
               or they default to 1.0 / array_size if unknown.

    Args:
        file_path:     path to the file
        fmt:           "spm" or "npy"
        pixel_size_nm: nm/pixel — required for "npy", ignored for "spm"
        scan_size_nm:  full scan size in nm — required for "npy", ignored for "spm"

    Returns:
        AFMRawData with z_raw (float32, nm), pixel_size_nm, scan_size_nm
    """
    if fmt == "spm":
        scan_size, px_size, z = _read_nanoscope_z(file_path)
        return AFMRawData(z_raw=z, pixel_size_nm=px_size, scan_size_nm=scan_size)

    if fmt == "npy":
        z = np.load(file_path).astype(np.float32)
        return AFMRawData(
            z_raw=z,
            pixel_size_nm=pixel_size_nm or 1.0,
            scan_size_nm=scan_size_nm or float(z.shape[0]),
        )

    raise ValueError(f"Unsupported format: {fmt}")


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
        raise FileNotFoundError(f"Could not read image: {file_path}")

    return MicroscopyData(
        image=image,
        nm_per_pixel=nm_per_pixel,
        modality=modality,
    )
