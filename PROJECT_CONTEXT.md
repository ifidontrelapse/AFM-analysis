# AFM Analysis — Project Context and Architecture

> Machine-oriented project documentation. Treat the repository source code as the source of truth and use this file as a map of the current implementation, interfaces, assumptions, and known gaps.

## 1. Project identity

- **Name:** `afm-analysis`
- **Version:** `0.1.0`
- **Purpose:** detect, segment, and measure nanoparticles in microscopy data, with the main scientific workflow designed for AFM height maps.
- **Primary language:** Python 3.12+.
- **Web client:** React 18 + TypeScript + Vite + Tailwind CSS.
- **License:** Apache-2.0 (`LICENSE`).
- **Current repository state:** the Python analysis library exists. There is no HTTP backend and no web client — the React client was deleted in ADR-0012, and the product is a Linux desktop application (ADR-0002).

The central AFM workflow is:

```text
raw AFM file
  -> load_afm
  -> plane flattening
  -> line flattening
  -> substrate estimation
  -> particle-positive map (z_result)
  -> LoG or YOLO detection
  -> optional SAM2 segmentation OR circular baseline measurement
  -> PipelineResult / pandas.DataFrame / plots
```

The project also contains an image-only SEM/TEM path. SEM/TEM data has no height map, so it can produce geometric mask measurements but cannot produce AFM height measurements.

## 2. How to read this document

For implementation tasks, inspect the relevant source files before changing this document. The most important contracts are:

1. `src/types.py` — shared Python dataclasses and configuration literals.
2. `src/preprocessing_pipeline.py` — standard AFM preprocessing entry point.
3. `src/pipeline.py` — detector/mode orchestration.
4. `src/detection/` and `src/segmentation.py` — model and algorithm implementations.

Do not infer that a feature exists only because it is mentioned in `README.md`, `project.md`, a notebook, or this file. Verify the implementation in `src/`. This document is refreshed in M2-T16 and lags the tree until then.

## 3. Repository map

```text
AFM-analysis/
├── src/                              # Python analysis library
│   ├── __init__.py                   # small public API re-export
│   ├── types.py                      # dataclass dependency root
│   ├── afm_io.py                     # SPM and NPY loading; SEM/TEM image loading
│   ├── preprocess.py                 # AFM flattening and substrate estimation
│   ├── preprocessing_pipeline.py     # load + preprocess orchestration
│   ├── detection/
│   │   ├── __init__.py               # detector re-exports
│   │   ├── base.py                   # BaseDetector and blob conversion
│   │   ├── log_detector.py            # classical LoG detector
│   │   └── yolo_detector.py           # YOLOv8 detector, tiled/direct backends
│   ├── segmentation.py                # SAM2 prompts, masks, and measurements
│   ├── measure.py                     # circular baseline and mask geometry
│   ├── pipeline.py                    # full detector/mode dispatcher
│   └── visualization.py               # matplotlib plots and interactive viewer
├── frontend/                         # independent React/Vite client
│   ├── src/api/client.ts              # POST /analyze client
│   ├── src/types/pipeline.ts          # TypeScript API types
│   ├── src/pages/AnalyzePage.tsx      # page state and layout
│   ├── src/components/                # upload, config, result, stats, histogram UI
│   └── package.json                   # frontend scripts/dependencies
├── configs/sam2_hiera_b+.yaml         # SAM2 model configuration
├── checkpoints/                       # local model weights; ignored by git
├── data/                              # local raw/preprocessed data; ignored by git
├── dataset/                           # local generated dataset; ignored by git
├── images/                            # committed example figures
├── *.ipynb                            # notebooks and experiments
├── pyproject.toml                     # Python metadata and uv dependencies
├── uv.lock                            # Python lock file
├── pytest.ini                         # adds . and src to Python path
├── README.md                          # user-facing overview; partly stale
├── project.md                         # earlier architecture notes; partly stale
└── PROJECT_CONTEXT.md                 # this document
```

`frontend/node_modules/`, Python caches, local data, and model checkpoints are runtime/development artifacts. They are not architectural source files. The current working tree also contains an existing staged `yolov8s-world.pt` and an untracked root `package-lock.json`; preserve unrelated user changes when modifying the project.

## 4. Layered architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│ React frontend (frontend/)                                     │
│ upload file -> select modality/detector/mode -> POST /analyze  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP contract is planned, not implemented here
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Missing adapter/backend                                        │
│ expected responsibility: parse multipart file + JSON config,   │
│ call Python pipeline, serialize result and preview image        │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Python domain core (src/)                                      │
│ I/O -> preprocessing -> detection -> segmentation/measurement  │
│ -> PipelineResult + plots                                      │
└─────────────────────────────────────────────────────────────────┘
```

The Python core is not a web service. It is a synchronous, in-process library. The model objects are passed into functions directly; there is no dependency injection container, task queue, persistence layer, database, or application server.

### Dependency direction

```text
src/types.py
    ↑
afm_io.py, preprocess.py, measure.py, segmentation.py, detection/*
    ↑
preprocessing_pipeline.py, pipeline.py
    ↑
visualization.py, notebooks, external callers
```

`src.types` is intentionally the dependency root and imports no other `src` module. `src.pipeline` imports detection, segmentation, and measurement code. `src.visualization` is a consumer of `PipelineResult`, not a processing stage required by the core.

## 5. Core data model

All arrays are expected to be two-dimensional image-like arrays unless stated otherwise. AFM height values are represented in nanometres after file calibration.

### `AFMRawData`

Defined in `src/types.py`.

```python
AFMRawData(
    z_raw: np.ndarray,        # float32 height map, nm
    pixel_size_nm: float,     # nm per pixel
    scan_size_nm: float,      # full scan width/height, nm
)
```

### `PreprocessingResult`

Produced by `run_preprocessing()`.

```python
PreprocessingResult(
    z_raw: np.ndarray,          # calibrated raw map
    z_flat: np.ndarray,         # plane- and line-flattened map
    z_result: np.ndarray,       # z_flat - substrate; particles should be positive
    substrate: np.ndarray,      # estimated substrate surface
    pixel_size_nm: float,
    scan_size_nm: float,
    sizes: dict,                # Otsu-derived particle radius statistics
    opening_radius: int,        # final morphological-opening radius in pixels
)
```

`z_result` is the image passed to the AFM detector and is also the default image converted to RGB for SAM2.

### `MicroscopyData`

Used for SEM/TEM input.

```python
MicroscopyData(
    image: np.ndarray,                     # grayscale image
    nm_per_pixel: float | None,             # optional physical scale
    modality: Literal["sem", "tem"],
)
```

No preprocessing is applied to SEM/TEM data by `run_pipeline`. Geometry is derived from segmentation masks.

### `Detection`

Detector-neutral particle representation:

```python
Detection(
    x_px: float,
    y_px: float,
    radius_px: float,
    radius_nm: float,
    confidence: float = 1.0,
    bbox: tuple[int, int, int, int] = (),  # x1, y1, x2, y2
)
```

Coordinates are image coordinates in pixels. LoG detections have a synthetic square bounding box derived from their radius. YOLO detections use the model box. LoG currently leaves `confidence` at its default `1.0`; the YOLO implementation currently does not copy model confidence scores into `Detection`.

### `PipelineConfig`

Defined in `src/types.py`. Current Python fields:

| Group | Fields and defaults |
|---|---|
| Selection | `detector="log"`, `mode="segment"` |
| LoG | `log_overlap=0.3`, `log_percentile=20.0`, `log_threshold=None` |
| YOLO | `yolo_model_path="./checkpoints/best12x.pt"`, `yolo_use_tiling=True`, `yolo_conf=0.5` |
| SAM2 | `sam2_outer_ring_px=5`, `sam2_inner_erode_px=2` |
| Circular baseline | `measure_outer_px=5`, `measure_inner_erode_px=3` |

The Python config does **not** contain a `modality` field. Modality is inferred from whether `run_pipeline` receives `PreprocessingResult` (`afm`) or `MicroscopyData` (`sem`/`tem`).

### `PipelineResult`

```python
PipelineResult(
    detections: list[Detection],
    masks: list[dict],              # empty in detect and baseline modes
    measurements: pd.DataFrame,     # empty in detect mode
    pixel_size_nm: float | None,
    detector_name: str,
    mode: str,
    modality: str,                  # afm | sem | tem
)
```

The `masks` dictionaries contain NumPy boolean masks and model scores, so the object is not directly JSON serializable. The `measurements` value is a pandas DataFrame and also needs explicit serialization for an HTTP API.

## 6. AFM preprocessing pipeline

The standard entry point is:

```python
from src.preprocessing_pipeline import run_preprocessing

pre = run_preprocessing("scan.spm", fmt="spm")
```

Implementation chain:

```text
load_afm
  -> flatten_plane
  -> flatten_lines(poly_order=1)
  -> build_substrate_map
  -> PreprocessingResult
```

### 6.1 Loading: `src/afm_io.py`

`load_afm(file_path, fmt, pixel_size_nm=None, scan_size_nm=None)` supports only:

- `fmt="spm"`: custom Bruker Nanoscope parser.
- `fmt="npy"`: `np.load`, converted to `float32`; metadata is supplied by the caller or defaults to `pixel_size_nm=1.0` and `scan_size_nm=z.shape[0]`.

For SPM, `_read_nanoscope_z`:

1. Reads the first 65,536 bytes and decodes the header as Latin-1.
2. Splits at `0x1A` and searches `\\*Ciao image list` blocks.
3. Chooses a block containing `"Height"` when available.
4. Extracts data offset, data length, samples/line, number of lines, and bytes/pixel.
5. Extracts `@2:Z scale` in volts and `@Sens. Zsens` in nm/V.
6. Reads signed 16-bit data for 2 bytes/pixel, otherwise signed 32-bit data.
7. Calibrates raw values with `z_scale_v * nm_per_v / 65536` and reshapes to `(lines, samples)`.
8. Extracts scan size and computes `pixel_size_nm = scan_size_nm / samples`.

`load_microscopy_image(path, modality, nm_per_pixel=None)` reads a grayscale image with OpenCV and returns `MicroscopyData`. Supported modality literals are `"sem"` and `"tem"`.

### 6.2 Flattening: `src/preprocess.py`

- `flatten_plane(z)` fits `z = ax + by + c` using least squares and subtracts the plane.
- `flatten_lines(z, poly_order=1)` fits and subtracts a polynomial independently for every row. The default is linear detrending.

These functions preserve the array shape and operate on the numeric map; they do not change the physical pixel scale.

### 6.3 Substrate estimation

`get_substrate_map(z, radius_px)` applies grayscale morphological opening with a disk structuring element. The intended assumption is that `radius_px` is larger than the largest particle radius, allowing the opening to preserve the substrate and remove particle peaks.

`build_substrate_map(z, pixel_size_nm, min_size_nm=5, manual_radius_px=None)` has two paths:

```text
automatic path:
  rough threshold (median + std)
  -> rough connected-component radius
  -> rough opening
  -> Otsu connected-component radius statistics
  -> final opening radius = max(int(typical_radius_px * 2.5), 5)
  -> substrate, z_above = z - substrate

manual path:
  opening with manual_radius_px
  -> Otsu radius statistics
  -> substrate, z_above
```

`estimate_radius_otsu` thresholds with `threshold_otsu`, labels connected components, converts each component area to an equivalent circular radius, and removes radii below `min_size_pixel`. It returns `typical_radius_px`, `typical_radius_nm`, arrays of radii, object count, and the Otsu threshold.

The `sizes` dictionary is later used to determine the LoG sigma range.

## 7. Detection architecture

All detectors implement `BaseDetector.detect(image, pixel_size_nm) -> list[Detection]`. The pipeline chooses a detector using `PipelineConfig.detector`.

### 7.1 LoG detector

Files: `src/detection/base.py`, `src/detection/log_detector.py`.

`LogDetector` wraps the functional `detect_particles` implementation and retains the raw blob array in `last_blobs` for SAM2 or circular baseline measurement.

Raw LoG blob format:

```text
blobs.shape == (N, 4)
blobs[i] == [y_px, x_px, sigma_px, radius_nm]
radius_px = sigma_px * sqrt(2)
radius_nm = radius_px * pixel_size_nm
```

Algorithm:

1. Get particle radii from preprocessing, or estimate them with Otsu if `sizes=None`.
2. Convert radius range to sigma range:
   - `min_sigma = max(min(radii_px) / sqrt(2) * 0.5, 1.0)`
   - `max_sigma = max(max(radii_px) / sqrt(2) * 2.0, min_sigma * 2)`
3. If no explicit threshold is supplied, run a low-threshold discovery pass and choose the requested response percentile (`log_percentile`, default 20) as the adaptive threshold.
4. Normalize by `z_above.max()` and call `skimage.feature.blob_log` with 15 sigma values and the configured overlap.
5. Convert sigma to physical radius and remove circles crossing the image boundary.

There is also `estimate_log_threshold`, which estimates `3 * substrate_noise_std / z_max`, but `detect_particles` uses `estimate_log_threshold_adaptive` when the threshold is `None`.

### 7.2 YOLO detector

File: `src/detection/yolo_detector.py`.

`YoloDetector` prepares an input image by resizing to `640 x 640` by default, normalizing to `[0, 255]`, inverting it, and converting grayscale to RGB.

Backends:

| `use_tiling` | Backend | Intended use |
|---|---|---|
| `True` (default) | `patched_yolo_infer.MakeCropsDetectThem` + `CombineDetections` | Dense fields; sliding windows reduce tile-boundary misses |
| `False` | `ultralytics.YOLO` | Direct inference; simpler and usually faster for sparse fields |

Both backends scale boxes from model coordinates back to the original image shape. A box becomes a `Detection` whose centre is the box centre and whose radius is half of the smaller box dimension. The raw backend result is retained in `last_result` for external visualization.

Required local weights are normally under `checkpoints/`; the Python default is `./checkpoints/best12x.pt`. The repository also contains `yolov8s-world.pt`, but that file is not the default configured model.

## 8. Segmentation and measurement

### 8.1 SAM2 segmentation

File: `src/segmentation.py`.

SAM2 is deliberately isolated from the rest of the code so detection and baseline measurement can run without importing SAM2 internals. The caller must provide an initialized `SAM2ImagePredictor` when `PipelineConfig.mode == "segment"`.

Input conversion:

```text
float image
  -> percentile clipping at 99th percentile
  -> normalize to [0, 1]
  -> matplotlib "afmhot" colormap
  -> uint8 RGB
```

Prompt paths:

- `run_sam2_from_blobs`: LoG centre point plus a padded square box based on the LoG radius.
- `run_sam2_from_boxes`: YOLO box plus its centre point.

For each prompt, SAM2 requests three masks and selects the mask with the highest score. In AFM mode, the selected mask is eroded for peak extraction and expanded for a local substrate ring. In SEM/TEM mode, `measure_geometry_from_mask` computes object geometry.

The result list stores NumPy masks (`mask`, optionally `mask_inner` and `ring`), coordinates, and score. The DataFrame stores scalar measurement fields.

### 8.2 Circular baseline mode

File: `src/measure.py`.

`measure_all_baseline` is available only for AFM + LoG through `run_pipeline`.

For each LoG blob it creates a circular mask with `radius_px = sigma * sqrt(2)`, then:

```text
clean ring = dilation(dilation(particle, inner_erode_px), outer_px)
             - dilation(particle, inner_erode_px)
             intersected with substrate_mask

baseline = median(z_flat[clean ring])
height   = max(z_flat[particle mask]) - baseline
mean     = mean(z_flat[particle mask]) - baseline
```

If the clean ring contains fewer than `min_ring_px` pixels (default 5), the global substrate median is used. Measurements with non-positive height are discarded.

The baseline DataFrame normally contains:

| Column | Meaning |
|---|---|
| `particle_id` | index of the input blob |
| `x_px`, `y_px` | centre coordinates |
| `sigma_px`, `radius_nm` | LoG size estimates |
| `method` | currently `baseline_circle` |
| `height_nm`, `mean_nm` | height statistics above baseline |
| `baseline_nm` | ring or global baseline |
| `area_px`, `ring_px` | mask/ring pixel counts |
| `baseline_source` | `ring` or `global` |

### 8.3 SEM/TEM geometry

`measure_geometry_from_mask` uses `skimage.measure.regionprops` on a binary mask and returns:

- `area_px`
- `area_nm2` when `nm_per_pixel` is available
- equivalent `radius_px` and `radius_nm`
- circularity `4π * area / perimeter²`
- aspect ratio `major_axis_length / minor_axis_length`

## 9. Pipeline orchestration

Main entry point:

```python
from src.pipeline import run_pipeline
result = run_pipeline(data, cfg, predictor=None)
```

`data` must be a `PreprocessingResult` for AFM or a `MicroscopyData` for SEM/TEM.

### Execution matrix

> **This table documents `nanoscope/application/capabilities.py`; it does not define the
> rules.** Before M2-T10 the same matrix existed here, as `if` statements in
> `src/pipeline.py`, and hardcoded in the React client — where the audit found it had
> already drifted (D-19). One copy executes; this one is for reading. A test asserts the
> row count matches, which is the most a document can be held to.

| Input | Detector | Mode | Behavior |
|---|---|---|---|
| AFM | LoG | `detect` | preprocess is assumed complete; return detections only |
| AFM | YOLO | `detect` | return YOLO detections only |
| AFM | LoG | `baseline` | circular masks + ring baseline; no SAM2 |
| AFM | YOLO | `baseline` | rejected with `ValueError` |
| AFM | LoG/YOLO | `segment` | SAM2 prompted from LoG blobs or YOLO boxes |
| SEM/TEM | LoG/YOLO | `detect` | image detection only |
| SEM/TEM | LoG/YOLO | `segment` | SAM2 masks + geometry measurements |
| SEM/TEM | any | `baseline` | rejected with `ValueError` |

`mode="segment"` always requires a non-`None` predictor. The pipeline does not construct a predictor or load a checkpoint itself.

Since M2-T10 every rejection above is raised **before a detector is constructed**, so an
invalid combination costs nothing (audit D-14). The messages are unchanged.

`run_full_pipeline` was deleted in M2-T13: it only forwarded to `run_pipeline`, and nothing called it. Use `run_pipeline` directly.

## 10. Visualization and batch processing

File: `src/visualization.py`.

- `plot_afm`: renders a height map with physical nm axes and optional colorbar.
- `afm_viewer`: interactive matplotlib viewer with cursor readout and two-point height profile.
- `plot_detections`: renders LoG circles and centres.
- `plot_detections_histogram` and `plot_pipeline_result` were deleted in M2-T13 — no caller in code, tests or notebooks.
- `afm_to_rgb` and `overlay_masks` live in `src/segmentation.py` and are used for mask visualization.

`preprocess_batch.py` is a CLI that recursively finds files whose extensions are numeric (for example `.001`, `.002`), applies the SPM preprocessing sequence, and writes rendered `.jpg` maps while mirroring the input directory structure.

```bash
python preprocess_batch.py data/raw/ data/preprocessed/
```

It does not save a structured `PreprocessingResult`; it saves visualization images only.

## 11. Frontend architecture

The frontend is a standalone Vite application under `frontend/`. It has no router, state library, chart library, or UI component library. State is local React state in `AnalyzePage`.

```text
App
  -> AnalyzePage
       -> UploadZone
       -> ConfigPanel
       -> ResultViewer
       -> StatsPanel
       -> Histogram (0..2 instances)
       -> LoadingOverlay (during request)
```

### Frontend behavior

- `UploadZone` accepts `.spm`, `.npy`, `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`.
- `ConfigPanel` supports AFM/SEM/TEM, LoG/YOLO, and detect/baseline/segment.
- Baseline is disabled for SEM/TEM in the UI.
- LoG controls: percentile and overlap.
- YOLO controls: tiling and confidence.
- `ResultViewer` displays a server-provided base64 PNG and a particle count badge.
- `StatsPanel` computes means/medians client-side from returned detections/measurements.
- `Histogram` renders SVG bars and mean/median guide lines without a chart library.
- `LoadingOverlay` covers the page during `POST /analyze`.

### Frontend HTTP contract

`frontend/src/api/client.ts` sends:

```http
POST ${VITE_API_URL}/analyze
Content-Type: multipart/form-data

image=<uploaded file>
config=<JSON stringified PipelineConfig>
```

`VITE_API_URL` is currently `http://localhost:8000` in `frontend/.env` and is also the client fallback.

The frontend expects this JSON shape:

```ts
interface PipelineResult {
  detections: Detection[];
  masks_preview_b64: string;
  measurements: ParticleMeasurement[];
  pixel_size_nm: number | null;
  detector_name: "log" | "yolo";
  mode: "detect" | "baseline" | "segment";
  modality: "afm" | "sem" | "tem";
  particle_count: number;
}
```

The frontend config additionally contains `modality` and optional `nm_per_pixel`:

```ts
interface PipelineConfig {
  modality: "afm" | "sem" | "tem";
  detector: "log" | "yolo";
  mode: "detect" | "baseline" | "segment";
  nm_per_pixel?: number;
  log_overlap?: number;
  log_percentile?: number;
  yolo_use_tiling?: boolean;
  yolo_conf?: number;
}
```

The adapter/backend still needs to map this HTTP contract to the Python objects, serialize dataclasses/DataFrames/masks, render a PNG preview, and calculate `particle_count`.

## 12. Python and frontend dependencies

### Python

Declared in `pyproject.toml` and locked in `uv.lock`:

- NumPy, SciPy, pandas: arrays, numerical operations, and tabular results.
- scikit-image: Otsu thresholding, connected components, LoG, morphology, region properties.
- matplotlib and ipympl: static and interactive plots.
- OpenCV: grayscale image loading and YOLO input preparation.
- `pyspm`: declared dependency for SPM-related tooling, although the current loader is custom.
- PyTorch and torchvision: model runtime; configured from the CUDA 11.8 PyTorch index.
- `sam-2` from the SAM2 Git repository and `sam2`: segmentation runtime.
- `ultralytics`: direct YOLO inference.
- `patched-yolo-infer`: tiled YOLO inference.
- tqdm, requests, ipykernel: utility/notebook/runtime support.

### Frontend

- Runtime: `react`, `react-dom`.
- Build/tooling: Vite, TypeScript, `@vitejs/plugin-react`.
- Styling: Tailwind CSS, PostCSS, Autoprefixer.
- No backend dependency is declared in this repository.

## 13. Installation and common commands

### Python environment

```bash
uv sync
```

The project requires Python `>=3.12`. PyTorch is configured for the CUDA 11.8 package index in `pyproject.toml`; change the `tool.uv.sources` configuration if a CPU or another CUDA setup is required.

### Frontend

```bash
cd frontend
npm install
npm run dev
npm run build
npm run preview
```

The current frontend can build as a static client, but it cannot complete analysis without an HTTP service implementing `POST /analyze`.

### Library usage

AFM preprocessing plus baseline measurement:

```python
from src.afm_io import load_afm
from src.preprocess import flatten_plane, flatten_lines, build_substrate_map
from src.detection import detect_particles
from src.measure import measure_all_baseline

raw = load_afm("data/sample.spm", fmt="spm")
z_flat = flatten_lines(flatten_plane(raw.z_raw), poly_order=1)
substrate, z_result, opening_radius, sizes = build_substrate_map(
    z_flat, raw.pixel_size_nm, min_size_nm=5
)
blobs = detect_particles(z_result, raw.pixel_size_nm, sizes)
df = measure_all_baseline(z_flat, z_result, blobs)
```

Preferred high-level preprocessing call:

```python
from src.preprocessing_pipeline import run_preprocessing
from src.pipeline import run_pipeline
from src.types import PipelineConfig

pre = run_preprocessing("data/sample.spm", fmt="spm")
result = run_pipeline(pre, PipelineConfig(detector="log", mode="baseline"))
```

For `mode="segment"`, initialize a compatible `SAM2ImagePredictor` separately and pass it as `predictor`.

## 14. Tests and quality gates

Current test coverage is minimal. `tests/test_io.py` contains a smoke test for SPM loading but does not assert output values and references a path that may not exist in a clean checkout. There are no unit tests covering preprocessing, detectors, SAM2, measurement, pipeline dispatch, frontend behavior, or HTTP serialization.

Configured checks:

```bash
pytest
ruff check .
```

Frontend type/build check:

```bash
cd frontend
npm run build
```

When changing numerical code, add deterministic tests with small synthetic arrays before relying on notebook output. When changing the API boundary, test both the serialized backend shape and the TypeScript consumer shape.

## 15. Known implementation gaps and risks

These are observations from the current source, not proposed behavior.

### High priority: frontend/backend contract is incomplete

- No FastAPI, Flask, or other server module exists.
- Python `PipelineResult` does not have `masks_preview_b64` or `particle_count`.
- Python `PipelineConfig` does not have `modality` or `nm_per_pixel`.
- Python masks contain NumPy arrays and measurements are a pandas DataFrame; neither is directly JSON serializable.
- The frontend currently assumes the server has already rendered the result preview.

### High priority: preprocessing manual-radius path

In `build_substrate_map`, the `manual_radius_px is not None` branch computes the substrate and sizes but does not assign `opening_radius` before the final return statement. Calling this branch can raise `UnboundLocalError`.

### Numerical edge cases

- LoG normalization divides by `z_above.max()`. A zero or non-positive map can produce invalid values or unstable thresholding.
- Otsu sizing can leave an empty radius array after minimum-size filtering even when connected components exist.
- The automatic rough-radius fallback prints a warning and uses approximately 1% of image width; this is a heuristic, not a calibrated estimate.
- Morphological opening assumes its radius exceeds the largest particle radius. An incorrect radius changes the physical meaning of `z_result`.
- Local ring measurement falls back to a global baseline when the ring is too small, which can hide local substrate gradients.
- `estimate_rough_radius` is annotated as returning `int` but can return a float after multiplying by its scale.
- The SPM parser assumes the relevant header fields and the image layout match the implemented regular expressions.

### Model and result semantics

- YOLO `Detection.confidence` is not populated from model scores.
- The YOLO input conversion casts to `uint8` before normalization; negative or non-8-bit height values may not be transformed as intended.
- SAM2 predictor initialization/checkpoint loading is outside the library and outside the frontend repository.
- `run_pipeline` assumes that any non-`PreprocessingResult` input is a valid `MicroscopyData` instance; explicit runtime validation is limited.
- The frontend uses an API field named `masks_preview_b64`, but the Python visualization layer currently returns matplotlib figures/arrays rather than a base64 encoder.

### Documentation drift

- `README.md` describes an older flat module layout and an older return convention for `load_afm`.
- `README.md` mentions modules and notebooks that are not all present at those paths.
- `project.md` is closer to the current architecture but still contains older function signatures in places.
- `plan.md` describes the intended frontend/backend integration, not an implemented backend.

When documentation conflicts with executable code, prefer `src/types.py`, function signatures, and actual control flow.

## 16. Guidance for future AI agents

Before implementing a change:

1. Identify whether the change belongs to the Python core, frontend, or the missing adapter boundary.
2. Read `src/types.py` and the complete call chain for the affected mode.
3. Preserve coordinate conventions: arrays use `[y, x]`, `Detection` exposes `x_px` and `y_px`, and boxes are `(x1, y1, x2, y2)`.
4. Preserve units: AFM `z_flat`, `z_result`, height, and radius values are in nanometres; pixel geometry is in pixels; SEM/TEM nanometre values require `nm_per_pixel`.
5. Keep model loading outside low-level numerical functions unless the existing interface explicitly requires it.
6. Do not serialize masks or DataFrames implicitly; define an explicit wire format at the HTTP boundary.
7. Add or update tests for edge cases before changing default numerical behavior.
8. Check repository status before editing and avoid touching local data, checkpoints, `node_modules`, or unrelated staged files.

Recommended order for adding the web backend:

```text
define canonical wire schemas
  -> implement file/config validation
  -> implement AFM vs SEM/TEM loading
  -> initialize/cache model predictors safely
  -> call run_preprocessing + run_pipeline
  -> serialize detections and measurements
  -> render PNG preview to base64
  -> add CORS/error handling
  -> add an integration test against frontend/src/types/pipeline.ts
```

## 17. Suggested next work, ordered by impact

1. Fix and test `build_substrate_map(manual_radius_px=...)`.
2. Add numerical input validation for empty, constant, negative-only, and malformed maps.
3. Define one canonical JSON schema shared by Python backend and TypeScript frontend.
4. Implement the missing `/analyze` adapter, including model lifecycle and preview encoding.
5. Add tests for preprocessing, LoG, baseline measurement, pipeline mode validation, and serialization.
6. Refresh `README.md` and `project.md` from this architecture document to remove stale claims.
7. Decide whether the legacy notebooks and `preprocess_batch.py` remain supported interfaces or should be marked experimental.
