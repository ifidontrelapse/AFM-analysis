# AFM Nanoparticle Analysis — Project Overview

## What the project does

Automated pipeline for detecting, segmenting, and measuring gold nanoparticles in Atomic Force Microscopy (AFM) height maps. Given a raw `.spm` (Bruker Nanoscope) or `.npy` file, the pipeline:

1. Loads and preprocesses the scan (plane/line flattening, substrate removal)
2. Detects particles — two backends: classical LoG or YOLOv8
3. Segments each particle — SAM2 (Segment Anything Model 2)
4. Measures physical properties — height in nm, baseline, mask area
5. Visualises results

---

## Module map

```
src/
├── types.py                  ← shared dataclasses (dependency root)
├── afm_io.py                 ← file I/O
├── preprocess.py             ← signal processing (flatten, substrate)
├── preprocessing_pipeline.py ← orchestrates I/O + preprocessing
├── detection/
│   ├── base.py               ← BaseDetector ABC
│   ├── log_detector.py       ← LoG detector
│   ├── yolo_detector.py      ← YOLOv8 detector
│   └── __init__.py           ← re-exports public API
├── segmentation.py           ← SAM2 segmentation
├── measure.py                ← height measurement (non-SAM2 path)
├── pipeline.py               ← full analysis orchestration
├── visualization.py          ← plotting
└── __init__.py               ← package public API
```

---

## Data types (`src/types.py`)

Dependency root — imports nothing from other `src/` modules.

| Dataclass | Fields | Produced by |
|-----------|--------|-------------|
| `PreprocessingResult` | `z_raw`, `z_flat`, `z_result`, `substrate`, `pixel_size_nm`, `scan_size_nm`, `sizes`, `opening_radius` | `run_preprocessing()` |
| `Detection` | `x_px`, `y_px`, `radius_px`, `radius_nm`, `confidence`, `bbox` | any detector |
| `PipelineConfig` | detector/mode selection + all tuning params | user |
| `PipelineResult` | `detections`, `masks`, `measurements`, `pixel_size_nm`, `detector_name`, `mode` | `run_pipeline()` |

---

## Module descriptions

### `src/afm_io.py`
Loads raw AFM files and returns `(scan_size_nm, pixel_size_nm, z: float32)`.

- **`load_afm(file_path, fmt)`** — dispatcher; supports `"spm"` and `"npy"`
- **`_read_nanoscope_z(path)`** — parses binary Bruker Nanoscope `.spm`: reads ASCII header, extracts `Data offset`, `Bytes/pixel`, `Z scale (V)`, `Zsens (nm/V)`, `Scan Size`. Converts raw int16/int32 to nm. Handles height-channel selection by searching for `"Height"` in the `\*Ciao image list` blocks.
- **`make_synthetic_afm()`** — stub, planned

No external AFM libraries used — fully custom parser.

---

### `src/preprocess.py`
Signal-processing layer. All functions operate on 2-D `float32` arrays in nm.

| Function | What it does |
|----------|-------------|
| `flatten_plane(z)` | Removes global tilt: fits a plane by least-squares (`scipy.linalg.lstsq`) and subtracts it |
| `flatten_lines(z, poly_order=1)` | Row-by-row linear detrend (`numpy.polyfit`) — removes per-line scanner drift |
| `get_substrate_map(z, radius_px)` | Morphological opening with `skimage.morphology.disk` — disk radius must exceed the largest particle radius. Returns the substrate surface |
| `estimate_rough_radius(z, ...)` | Quick radius guess: threshold at `median + std`, label blobs, take `sqrt(median_area / π) × scale` |
| `estimate_radius_otsu(z_above, ...)` | Otsu threshold on `z_above`, label connected regions, return median equivalent radius + full distribution |
| `build_substrate_map(z, pixel_size_nm, ...)` | Two-stage: rough radius → opening → Otsu sizing → final opening. Returns `(substrate, z_above, opening_radius, sizes)` |

`z_result = z_flat - substrate` is the key signal: it is zero on the substrate and positive on particles.

---

### `src/preprocessing_pipeline.py`
Thin orchestration wrapper.

**`run_preprocessing(file_path, fmt) → PreprocessingResult`**

Chain: `load_afm → flatten_plane → flatten_lines → build_substrate_map`

No parameters to tune — all defaults come from `preprocess.py`.

---

### `src/detection/base.py`
- **`BaseDetector`** — abstract base class. Single abstract method: `detect(z_above, pixel_size_nm) → list[Detection]`. Static helper `_blobs_to_detections` converts `(N, 4)` blob arrays to `Detection` objects.
- **`Detection`** — imported from `src.types` and re-exported.

---

### `src/detection/log_detector.py`
Laplacian of Gaussian particle detector.

**Helper functions (also public via `__init__.py`):**

| Function | Purpose |
|----------|---------|
| `estimate_log_params(sizes)` | Converts Otsu radii to `(min_sigma, max_sigma)` for `blob_log`. Range: `[r_min/√2 × 0.5, r_max/√2 × 2.0]` |
| `estimate_log_threshold(z_above)` | Conservative threshold: `3 × σ_noise / z_max` where noise is estimated from substrate pixels |
| `estimate_log_threshold_adaptive(z_above, params, percentile)` | Runs `blob_log` at threshold=0.01, collects peak responses of all blobs, returns `percentile`-th value. More robust across images |
| `detect_particles(z_above, ...)` | Full LoG detection: normalize → `blob_log` → append `radius_nm` → filter boundary blobs. Returns `(N, 4)` array `[y, x, sigma_px, radius_nm]` |
| `_filter_boundary_blobs(blobs, shape)` | Removes detections whose circle crosses the image boundary |

**`LogDetector(BaseDetector)`**
Stateful wrapper. Stores `_last_blobs` for downstream SAM2 calls that need `(N, 4)` format. If `sizes=None`, estimates them internally via Otsu.

---

### `src/detection/yolo_detector.py`
YOLOv8-based detector. Two backends:

| Mode | Library | Use case |
|------|---------|---------|
| `use_tiling=True` (default) | `patched_yolo_infer` (`MakeCropsDetectThem` + `CombineDetections`) | Dense particle fields — sliding-window approach avoids missed detections at tile boundaries |
| `use_tiling=False` | `ultralytics.YOLO` | Sparse fields or quick testing |

Image preparation: resize to `yolo_size×yolo_size`, normalize to `[0,255]`, invert (AFM particles appear as dark bumps on bright background after this transform), convert to RGB.

Boxes are scaled back from `yolo_size` to original image coordinates. Stored in `_last_result` for external visualisation.

---

### `src/segmentation.py`
SAM2-based segmentation. Takes an initialised `SAM2ImagePredictor`.

| Function | Prompt type | Input |
|----------|------------|-------|
| `run_sam2_from_blobs(predictor, z_flat, z_result, blobs, ...)` | point + box | `(N,4)` blob array from LoG |
| `run_sam2_from_boxes(predictor, z_flat, z_result, boxes_xyxy, ...)` | point + box | `(N,4)` boxes from YOLO |

Both functions:
1. Convert `z_result` to RGB (`afm_to_rgb`) and call `predictor.set_image`
2. For each particle: run SAM2 with best-of-3 mask selection
3. Build `mask_inner` (eroded) and `ring` (dilated, intersected with substrate mask)
4. Measure `peak = max(z_flat[mask_inner])`, `baseline = median(z_flat[ring])`
5. Return `(DataFrame, list[dict])` where each dict contains `mask`, `mask_inner`, `ring`, `score`

Helper functions: `afm_to_rgb(z)` — converts height map to uint8 RGB with percentile clipping; `overlay_masks(rgb, results)` — alpha-composite coloured masks for visualisation.

---

### `src/measure.py`
Non-SAM2 height measurement using circular masks from LoG radii.

| Function | Purpose |
|----------|---------|
| `create_circular_mask(shape, cy, cx, radius)` | Boolean disc mask |
| `get_clean_ring(mask_particle, substrate_mask, outer_px, inner_erode_px)` | Ring around particle, masked to substrate pixels only — removes neighbouring particles |
| `measure_height(z_flat, mask_particle, substrate_mask, ...)` | `height = max(z[mask]) - median(z[ring])`. Falls back to global baseline if ring is too small |
| `measure_all_baseline(z_flat, z_above, blobs, ...)` | Iterates over all blobs, builds circular masks, calls `measure_height`, returns DataFrame |

---

### `src/pipeline.py`
Full analysis orchestration. Assumes preprocessing is already done.

**`run_pipeline(z_flat, z_result, pixel_size_nm, cfg, sizes, predictor) → PipelineResult`**

```
cfg.detector == "log"   →  LogDetector.detect()   →  blobs (N,4)
cfg.detector == "yolo"  →  YoloDetector.detect()  →  boxes (N,4)

cfg.mode == "detect"    →  return detections only
cfg.mode == "segment"   →  run_sam2_from_blobs / run_sam2_from_boxes  →  measurements + masks
```

`PipelineConfig` and `PipelineResult` are defined in `src/types.py`.

---

### `src/visualization.py`

| Function | Description |
|----------|-------------|
| `plot_afm(ax, z, scan_size_nm)` | Single AFM panel with correct nm axis scale and colorbar |
| `afm_viewer(z, scan_size_nm)` | Interactive viewer: crosshair, coordinate readout, two-point height profile |
| `plot_detections(z_above, blobs, pixel_size_nm, axes)` | AFM image with LoG detection circles |
| `plot_detections_histogram(blobs, axes)` | Radius distribution histogram for LoG blobs |
| `plot_pipeline_result(result, z_result, scan_size_nm)` | Summary figure: 2 panels (detect mode) or 3 panels (segment mode) — image + radius histogram + height histogram |

---

## Full data flow

```
.spm / .npy file
       │
       ▼
  load_afm()
       │  scan_size_nm, pixel_size_nm, z_raw (float32, nm)
       ▼
  flatten_plane()  ──►  flatten_lines()
                                │  z_flat
                                ▼
                    build_substrate_map()
                                │  substrate, z_result, opening_radius, sizes
                                ▼
                 ┌──────────────┴──────────────┐
                 │                             │
           LogDetector                   YoloDetector
           .detect(z_result)             .detect(z_result)
                 │  list[Detection]            │  list[Detection]
                 │  .last_blobs (N,4)          │  .last_result
                 └──────────────┬──────────────┘
                                │
                    cfg.mode == "segment"?
                   YES ◄────────┴────────► NO → return PipelineResult
                    │
          run_sam2_from_blobs / _from_boxes
                    │  measurements (DataFrame), masks (list[dict])
                    ▼
              PipelineResult
                    │
          plot_pipeline_result()
```

---

## Dependency graph

```
src/types.py                       ← no src/ imports (root)
    ↑
src/afm_io.py
src/preprocess.py
src/measure.py
src/segmentation.py
src/detection/base.py
src/detection/log_detector.py
src/detection/yolo_detector.py
    ↑
src/preprocessing_pipeline.py
src/pipeline.py
    ↑
src/visualization.py
```

---

## External dependencies

| Library | Used for |
|---------|---------|
| `numpy` / `scipy` | Array math, LSQ plane fitting |
| `scikit-image` | `blob_log`, `threshold_otsu`, `label`, `regionprops`, morphological ops |
| `matplotlib` | All visualisation |
| `pandas` | Measurement DataFrames |
| `ultralytics` | YOLOv8 inference (non-tiled path) |
| `patched_yolo_infer` | Sliding-window YOLOv8 (tiled path) |
| `sam2` | SAM2ImagePredictor (segmentation path) |
| `cv2` (OpenCV) | Image resize/normalize for YOLO input |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/afm_gold_nanoparticles.ipynb` | Main interactive notebook — end-to-end pipeline demo |
| `notebooks/preprocessing.ipynb` | Preprocessing exploration |

Moved into `notebooks/` and stripped of outputs in M1-T09 (8.3 MB → 32 KB). `sam2.ipynb`
does not exist in the repository, and `main.ipynb` was a tracked 0-byte file that was not
valid JSON — deleted in M1-T09.
