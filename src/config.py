"""Configuration defaults for the AFM analysis pipeline.

Keep this separate so scripts/notebooks can import a single source of truth
and override only what they need.
"""

from __future__ import annotations

from typing import Any


def default_cfg() -> dict[str, Any]:
    """Return a dictionary of default pipeline parameters."""
    return {
        # File IO
        "file_path": "/home/matsu/AFM-analysis/data/2025/18 February/4.007",
        "file_format": "spm",  # "spm" | "ibw" | "gwy" | "npy"
        # Scan parameters
        "pixel_size_nm": 1.0,  # nm/px; prefer reading from metadata if available
        # Expected particle sizes
        "particle_min_nm": 3.0,
        "particle_max_nm": 80.0,
        "min_height_nm": 0.5,
        # LoG detector
        "log_threshold": 0.05,
        "log_overlap": 0.3,
        # Morphological opening (substrate map)
        "opening_radius_px": 40,
        # Ring baseline
        "ring_outer_px": 7,
        "ring_inner_erode_px": 2,
        # SAM 2
        "sam_checkpoint": "../sam2.1_hiera_base_plus.pt",
        "sam_config": "../sam2.1_hiera_b+.yaml",
        "device": "cuda",  # "cuda" or "cpu"
        # Z -> RGB colormap
        "colormap": "afmhot",
        "clip_percentile": 99.0,
    }
