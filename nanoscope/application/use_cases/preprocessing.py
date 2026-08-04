"""Load a raw AFM file and level it into arrays `run_pipeline` can use.

Moved from `src/preprocessing_pipeline.py` in M2-T15. `application`, for the same
reason as its neighbour: it sequences a loader from `infrastructure.storage` and
three steps from `core.science`, and owns none of them.

It had no caller in the repository, which is why the audit called it dead. It is
the documented preprocessing entry point in `README` and `Development.md`, and
M2-T13 kept it deliberately.
"""

from __future__ import annotations

from pathlib import Path

from nanoscope.core.entities import PreprocessingResult
from nanoscope.core.science.preprocessing import (
    build_substrate_map,
    flatten_lines,
    flatten_plane,
)
from nanoscope.infrastructure.storage import load_afm


def run_preprocessing(
    file_path: str | Path,
    fmt: str = "spm",
) -> PreprocessingResult:
    """
    Load and preprocess a raw AFM file.

    Steps:
        1. load_afm            — read file, extract z and pixel_size_nm
        2. flatten_plane       — remove global tilt (least-squares plane)
        3. flatten_lines       — row-by-row linear detrending
        4. build_substrate_map — morphological opening to estimate substrate,
                                 compute z_result = z_flat - substrate,
                                 estimate particle radii via Otsu

    All parameters use library defaults — no manual tuning required.

    Args:
        file_path: path to the AFM file
        fmt:       file format passed to load_afm ("spm" or "npy")

    Returns:
        PreprocessingResult with all arrays and metadata
    """
    raw = load_afm(str(file_path), fmt=fmt)

    z_plane = flatten_plane(raw.z_raw)
    z_flat = flatten_lines(z_plane)

    substrate, z_result, opening_radius, sizes = build_substrate_map(
        z_flat,
        raw.pixel_size_nm,
    )

    return PreprocessingResult(
        z_raw=raw.z_raw,
        z_flat=z_flat,
        z_result=z_result,
        substrate=substrate,
        pixel_size_nm=raw.pixel_size_nm,
        scan_size_nm=raw.scan_size_nm,
        sizes=sizes,
        opening_radius=opening_radius,
    )
