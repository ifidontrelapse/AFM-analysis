"""
AFM preprocessing pipeline.

Loads a raw AFM file and produces ready-to-use arrays for run_pipeline().

Typical usage:

    from src.preprocessing_pipeline import run_preprocessing

    pre = run_preprocessing("scan.spm", fmt="spm")
    # pre.z_flat, pre.z_result, pre.pixel_size_nm, pre.scan_size_nm, pre.sizes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.types import PreprocessingResult
from src.afm_io import load_afm
from src.preprocess import flatten_plane, flatten_lines, build_substrate_map


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
    # 1. Load
    scan_size_nm, pixel_size_nm, z_raw = load_afm(str(file_path), fmt=fmt)

    # 2. Flatten plane
    z_plane = flatten_plane(z_raw)

    # 3. Flatten lines
    z_flat = flatten_lines(z_plane)

    # 4. Build substrate map + estimate sizes
    substrate, z_result, opening_radius, sizes = build_substrate_map(
        z_flat,
        pixel_size_nm,
    )

    return PreprocessingResult(
        z_raw=z_raw,
        z_flat=z_flat,
        z_result=z_result,
        substrate=substrate,
        pixel_size_nm=pixel_size_nm,
        scan_size_nm=scan_size_nm,
        sizes=sizes,
        opening_radius=opening_radius,
    )
