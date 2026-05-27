"""
Shared dataclasses for the AFM nanoparticle analysis pipeline.

This module has no imports from other src/ modules — it is the dependency root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# ── I/O ───────────────────────────────────────────────────────────────────────

@dataclass
class AFMRawData:
    """Raw output of load_afm — before any preprocessing."""
    z_raw:         np.ndarray
    pixel_size_nm: float
    scan_size_nm:  float


# ── Preprocessing ─────────────────────────────────────────────────────────────

@dataclass
class PreprocessingResult:
    z_raw:          np.ndarray   # raw Z-map straight from the file
    z_flat:         np.ndarray   # after plane + line flattening
    z_result:       np.ndarray   # z_flat - substrate (particles above substrate)
    substrate:      np.ndarray   # estimated substrate surface
    pixel_size_nm:  float        # nm/pixel
    scan_size_nm:   float        # full scan size in nm
    sizes:          dict         # output of estimate_radius_otsu
    opening_radius: int          # morphological opening radius used


# ── SEM / TEM ─────────────────────────────────────────────────────────────────

@dataclass
class MicroscopyData:
    """
    Image data for SEM or TEM — no preprocessing, no height map.
    Geometry (area, radius, circularity) is derived from segmentation masks.
    """
    image:        np.ndarray
    nm_per_pixel: float | None             # None if physical scale is unknown
    modality:     Literal["sem", "tem"]


# ── Detection ─────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single particle detection result."""
    x_px:       float
    y_px:       float
    radius_px:  float
    radius_nm:  float
    confidence: float = 1.0
    bbox: tuple[int, int, int, int] = field(default_factory=tuple)  # x1,y1,x2,y2


# ── Pipeline ──────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    detector: Literal["log", "yolo"] = "log"
    mode:     Literal["detect", "baseline", "segment"] = "segment"

    # LogDetector params
    log_overlap:    float        = 0.3
    log_percentile: float        = 20.0
    log_threshold:  float | None = None

    # YoloDetector params
    yolo_model_path: str   = "./checkpoints/best12x.pt"
    yolo_use_tiling: bool  = True
    yolo_conf:       float = 0.5

    # SAM2 params
    sam2_outer_ring_px:  int = 5
    sam2_inner_erode_px: int = 2

    # Measure params
    measure_outer_px:       int = 5
    measure_inner_erode_px: int = 3


@dataclass
class PipelineResult:
    detections:    list[Detection]
    masks:         list[dict]     # empty if mode="detect"
    measurements:  pd.DataFrame   # empty if mode="detect"
    pixel_size_nm: float | None   # None if nm_per_pixel was unknown
    detector_name: str
    mode:          str
    modality:      str            # "afm" | "sem" | "tem"
