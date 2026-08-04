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

1. `nanoscope/core/entities/` — shared Python dataclasses and configuration literals.
2. `nanoscope/application/use_cases/preprocessing.py` — standard AFM preprocessing entry point.
3. `nanoscope/application/use_cases/pipeline.py` — detector/mode orchestration.
4. `src/detection/` and `nanoscope/infrastructure/models/sam2.py` — model and algorithm implementations.

Do not infer that a feature exists only because it is mentioned in `README.md`, `project.md`, a notebook, or this file. Verify the implementation in `nanoscope/`. This document was refreshed to the current tree in **M2-T16 (2026-08-04)**; `docs/STATE.md` is the one that is updated every session.

## 3. Repository map

```text
AFM-analysis/
├── nanoscope/                          # the library — Clean Architecture, ADR-0001/0011
│   ├── py.typed                        # the package is typed
│   ├── app/                            # composition root (empty until M5)
│   ├── core/                           # DOMAIN. No Qt, no torch, no matplotlib, no I/O
│   │   ├── entities/                   # AFMRawData, MicroscopyData, PreprocessingResult,
│   │   │                               #   Detection, PipelineConfig, PipelineResult
│   │   ├── values/                     # Modality, Polarity, PixelScale, DeviceKind
│   │   ├── ports/                      # Detector (the only port with implementations)
│   │   └── science/                    # the preserved numerical core
│   │       ├── io/nanoscope_spm.py     # SPM header parsing and calibration
│   │       ├── preprocessing/          # flatten.py (levelling), substrate.py (opening/Otsu)
│   │       ├── detection/              # base.py (ABC), log.py (LoG — pure NumPy)
│   │       └── measurement/            # height.py (AFM), geometry.py (any modality)
│   ├── application/
│   │   ├── capabilities.py             # THE execution matrix, validated before inference
│   │   └── use_cases/                  # pipeline.py, preprocessing.py
│   ├── infrastructure/                 # everything that touches a file, a GPU or a framework
│   │   ├── storage/loaders.py          # load_afm, load_microscopy_image
│   │   ├── models/                     # yolo.py, sam2.py — heavy imports, function-local
│   │   └── imaging/                    # colormap.py, plots.py (matplotlib)
│   ├── gui/                            # PySide6 (M5)
│   └── resources/                      # assets, a package so importlib.resources finds them
├── tests/
│   ├── unit/                           # afm_io, values, ports, capabilities, logging,
│   │                                   #   import_graph — 118 tests
│   └── characterization/               # the golden: phantoms.py, capture.py, golden/
├── docs/                               # STATE, Progress, TASKS, Roadmap, ADR/, audit/
├── notebooks/                          # experiments; nothing may import them
├── configs/sam2_hiera_b+.yaml          # SAM2 model configuration
├── checkpoints/ data/ dataset/         # local, git-ignored
├── images/                             # committed example figures
├── Makefile                            # `make check` — the whole gate, and what CI calls
├── pyproject.toml                      # metadata, dependencies, and every tool's config
└── uv.lock
```

**Gone, and not coming back:** `src/` (deleted M2-T15 — the last shims), `frontend/` and
`preprocess_batch.py` (deleted by ADR-0012), `pytest.ini` (folded into `pyproject.toml` in
M1-T05). There is no `pythonpath` entry anywhere: the package is installed.

## 4. Layered architecture

```text
        ┌──────────────────────────────────────────────────────────┐
        │ app/           composition root — bootstrap, wiring (M5) │
        └──────────────────────────┬───────────────────────────────┘
        ┌──────────────┐           │           ┌────────────────────┐
        │ gui/  (M5)   │──────────▶│◀──────────│ infrastructure/    │
        │ PySide6      │  application/         │ storage, models,   │
        └──────────────┘  use cases,           │ imaging            │
                          capabilities         └─────────┬──────────┘
                                  │                      │ implements
                                  ▼                      ▼
                    ┌──────────────────────────────────────────────┐
                    │ core/  entities · values · ports · science    │
                    │ pure Python + NumPy                          │
                    └──────────────────────────────────────────────┘
```

**The dependency rule points inward, and it is a test, not a diagram.**
`tests/unit/test_import_graph.py` parses every module under `core/` and fails if one
imports `application`, `infrastructure` or `gui`; a second check runs in a subprocess and
fails if importing the domain loads torch, ultralytics, sam2, matplotlib, PySide6, cv2 or
pandas. Both were proven to fail on a real violation (M2-T09).

The library is synchronous and in-process. There is no server, task queue, database or DI
container yet — `app/` is where they will be wired (M5, M6).

### Dependency direction

```text
core/entities, core/values          ← imports nothing of ours
    ↑
core/ports                          ← entities only
    ↑
core/science                        ← entities; never ports, never outward
    ↑
infrastructure/*                    ← core.science, core.entities
    ↑
application/use_cases               ← core.science, infrastructure, capabilities
    ↑
gui/, notebooks, external callers
```

Measured, not asserted: `import nanoscope.core.entities` loads **185 modules in 0.07 s**,
of which 141 are numpy. Before M2-T09 the equivalent import cost 626 modules and pulled in
matplotlib and pandas.

**One honest exception.** `application/use_cases/pipeline.py` imports `YoloDetector` from
`infrastructure` by name instead of receiving a `Detector`. That is the `if/elif` dispatch
the port exists to remove, and M4 removes it when a container exists to do the choosing.
mypy already reports it as an error rather than letting it pass quietly.

## 5. Core data model

All arrays are expected to be two-dimensional image-like arrays unless stated otherwise. AFM height values are represented in nanometres after file calibration.

### `AFMRawData`

Defined in `nanoscope/core/entities/`.

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

Defined in `nanoscope/core/entities/`. Current Python fields:

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
from nanoscope.application.use_cases import run_preprocessing

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

### 6.1 Loading: `nanoscope/core/science/io/nanoscope_spm.py` + `nanoscope/infrastructure/storage/loaders.py`

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

### 6.2 Flattening: `nanoscope/core/science/preprocessing/`

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

Files: `nanoscope/core/science/detection/base.py`, `nanoscope/core/science/detection/log.py`.

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

File: `nanoscope/infrastructure/models/yolo.py`.

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

File: `nanoscope/infrastructure/models/sam2.py`.

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

File: `nanoscope/core/science/measurement/`.

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
from nanoscope.application.use_cases import run_pipeline
result = run_pipeline(data, cfg, predictor=None)
```

`data` must be a `PreprocessingResult` for AFM or a `MicroscopyData` for SEM/TEM.

### Execution matrix

> **This table documents `nanoscope/application/capabilities.py`; it does not define the
> rules.** Before M2-T10 the same matrix existed here, as `if` statements in
> `nanoscope/application/use_cases/pipeline.py`, and hardcoded in the React client — where the audit found it had
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

File: `nanoscope/infrastructure/imaging/plots.py`.

- `plot_afm`: renders a height map with physical nm axes and optional colorbar.
- `afm_viewer`: interactive matplotlib viewer with cursor readout and two-point height profile.
- `plot_detections`: renders LoG circles and centres.
- `plot_detections_histogram` and `plot_pipeline_result` were deleted in M2-T13 — no caller in code, tests or notebooks.
- `afm_to_rgb` and `overlay_masks` live in `nanoscope/infrastructure/imaging/colormap.py` (M2-T07 — neither has anything to do with SAM2).

`preprocess_batch.py` was **deleted** by ADR-0012. It unpacked `AFMRawData` as a 3-tuple
and therefore failed on every file it was given (D-02); it had been broken since the commit
that introduced the dataclass. Batch processing returns as a use case when there is an
application to run it from.

## 11. The web client (removed)

The React/Vite client under `frontend/` was **deleted** by **ADR-0012** (2026-08-04), which
supersedes ADR-0007. It targeted a `POST /analyze` backend that was never written, and the
product direction is a Qt6 desktop application (ADR-0002). It also hardcoded its own copy
of the execution matrix, which the audit found had already drifted from the Python one
(D-19) — that duplication is what M2-T10 finally removed.

The code remains in git history. Nothing in this document describes it as current.

## 12. Dependencies

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

### Development and CI

Declared in `[dependency-groups]`: **pytest 9.1.1, pytest-cov 7.1.0, ruff 0.16.1,
mypy 2.3.0, pre-commit 4.6.1** (M1-T02).

There is a second group, `ci`, and it is deliberately smaller than the runtime one:
numpy, scipy, scikit-image, pandas, matplotlib, headless OpenCV and the dev tools —
**no torch, ultralytics, sam2 or patched-yolo-infer**. Every heavy import in the library is
function-local, so the suite never reaches them, and CI has no reason to resolve a CUDA
wheel or clone SAM2. A step in the workflow asserts this rather than trusting it.

## 13. Installation and common commands

### Python environment

```bash
uv sync
```

The project requires Python `>=3.12`. PyTorch is configured for the CUDA 11.8 package index in `pyproject.toml`; change the `tool.uv.sources` configuration if a CPU or another CUDA setup is required.


The current frontend can build as a static client, but it cannot complete analysis without an HTTP service implementing `POST /analyze`.

### Library usage

AFM preprocessing plus baseline measurement:

```python
from nanoscope.core.science.detection import detect_particles
from nanoscope.core.science.measurement import measure_all_baseline
from nanoscope.core.science.preprocessing import (
    build_substrate_map,
    flatten_lines,
    flatten_plane,
)
from nanoscope.infrastructure.storage import load_afm

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
from nanoscope.application.use_cases import run_preprocessing
from nanoscope.application.use_cases import run_pipeline
from nanoscope.core.entities import PipelineConfig

pre = run_preprocessing("data/sample.spm", fmt="spm")
result = run_pipeline(pre, PipelineConfig(detector="log", mode="baseline"))
```

For `mode="segment"`, initialize a compatible `SAM2ImagePredictor` separately and pass it as `predictor`.

## 14. Tests and quality gates

**One command, and it is the one CI runs** (M1-T10):

```bash
make check      # ruff format --check -> ruff check -> pytest, stopping at the first failure
make fast       # everything except the golden (~1 s) — the inner loop, not a merge gate
```

`make` on its own lists the targets. `.github/workflows/ci.yml` invokes those same targets
rather than repeating the commands, so the local gate and CI cannot describe different
things.

**118 tests**, of which the important one is the characterization golden: 8 seeded phantoms,
compared at `rtol=1e-6`, covering preprocessing, LoG detection, baseline measurement, YOLO
input preparation, degenerate inputs and the serialization contract. It runs inside
`pytest` (M1-T05) rather than by discipline, and every one of M2's sixteen relocations had
to pass it.

The golden records error *types and messages* as well as numbers, which is why translating
one Russian exception message in M2-T12 showed up as a declared four-line change — and why
**B-058** matters: it is pinned to CPython's minor version, so a Python upgrade reads as
drift. That needs an ADR before anyone upgrades.

Also enforced, and each proven to fail on a real violation:

| Check | File |
|---|---|
| `core` imports nothing from an outer layer; the domain loads no torch/matplotlib/Qt | `tests/unit/test_import_graph.py` |
| No `print` in library code | `tests/unit/test_logging.py` |
| Both detectors satisfy the `Detector` port | `tests/unit/test_ports.py` |
| Invalid requests are rejected *before* a detector is constructed | `tests/unit/test_capabilities.py` |

Nine pre-commit hooks run on every commit (M1-T07); `pytest` and mypy are deliberately not
among them — the golden alone takes 190 s, and a hook that slow is a hook people bypass.

mypy reports **20 errors**, all inside `nanoscope`, all inherited with the moved code and
none silenced. New code is strict from its first line; the moved scientific core runs at
default strictness under a declared override that shrinks as M3 lands.

## 15. Known implementation gaps and risks

These are observations from the current source, not proposed behavior.

> **These are tracked, not just noted.** Every item below is a numbered defect in
> `docs/audit/2026-07-28-baseline-audit.md` with a task in `docs/TASKS.md` §M3. M2 moved the
> code without fixing any of them, on purpose: a move that also changes a number makes a red
> golden ambiguous.

### Resolved since this document was first written

- **The frontend/backend contract** is not incomplete, it is gone — ADR-0012 deleted the
  client. Serialization returns as a concern in M6, against a desktop application.
- **Validation ran after inference** (D-14): fixed in M2-T10. Every rejection now happens
  before a detector is constructed.
- **13 `print` calls and 197 Russian lines** in library code: fixed in M2-T11 and M2-T12.
- **Five import cycles** (D-18): fixed in M2-T09, and a test refuses new ones.

### Critical: the manual-radius path fails on every call

In `build_substrate_map`, the `manual_radius_px is not None` branch never assigns
`opening_radius` before returning it — `UnboundLocalError`, 100% of calls. **D-01, M3-T01**,
and the golden already records the exception, so the fix will show as a declared change.

### Numerical edge cases

- LoG normalization divides by `z_above.max()`. A zero or non-positive map can produce invalid values or unstable thresholding.
- Otsu sizing can leave an empty radius array after minimum-size filtering even when connected components exist.
- The automatic rough-radius fallback logs a warning (M2-T11) and uses approximately 1% of image width; this is a heuristic, not a calibrated estimate.
- Morphological opening assumes its radius exceeds the largest particle radius. An incorrect radius changes the physical meaning of `z_result`.
- Local ring measurement falls back to a global baseline when the ring is too small, which can hide local substrate gradients.
- `estimate_rough_radius` is annotated as returning `int` but can return a float after multiplying by its scale. mypy reports this; it is not silenced.
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

When documentation conflicts with executable code, prefer `nanoscope/core/entities/`, function signatures, and actual control flow.

## 16. Guidance for future AI agents

**Read `docs/STATE.md` first.** It is updated every session and names the current task; this
document is the map, not the log.

Before implementing a change:

1. **Identify the layer.** Numerical science → `core/science`. Orchestration →
   `application/use_cases`. Anything touching a file, a GPU or a framework →
   `infrastructure`. If a change to `core` needs something from an outer layer, that is a
   port, and `tests/unit/test_import_graph.py` will refuse the shortcut.
2. Read `nanoscope/core/entities/` and the complete call chain for the affected mode.
3. **Preserve coordinate conventions:** arrays are `[y, x]`, `Detection` exposes `x_px` and
   `y_px`, boxes are `(x1, y1, x2, y2)`.
4. **Preserve units:** AFM `z_flat`, `z_result`, height and radius are nanometres; pixel
   geometry is pixels; SEM/TEM nanometre values require `nm_per_pixel`.
5. Keep model loading and heavy imports out of low-level numerical functions. CI installs
   no torch — a module-level `import torch` turns the job red, which is the point.
6. **Run `make check` before claiming anything works.** If the golden moves, the change is
   numerical whether or not it was meant to be.
7. **A numerical change gets its own commit, its own ADR, its own golden update, and a
   quantified before/after delta in `docs/Progress.md`.** Never bundled — that rule is what
   makes the golden a gate instead of a formality (ADR-0008, ADR-0010).
8. Do not touch local data, checkpoints, or unrelated staged files.

## 17. Suggested next work, ordered by impact

M2 is complete: the domain is extracted, the layout is enforced by tests, and every golden
number survived. The order below is `docs/Roadmap.md`'s, not a separate opinion.

1. **M3-T01** — fix `build_substrate_map(manual_radius_px=...)` (D-01, `UnboundLocalError`
   on 100% of calls). First numerical fix in the project.
2. **M3-T03** — YOLO input: normalise *then* cast. Only 12.6% of the dynamic range
   currently survives the conversion (D-03).
3. **Three defects need an operator decision before they can be fixed**, and each blocks a
   task: `min_size_nm` semantics (B2/M3-T02), opening-radius rounding (B4/M3-T09),
   detection polarity for TEM (B3/M3-T10). They are physics questions, not engineering ones.
4. **B-058** — the golden compares CPython exception text, so a Python upgrade reads as
   drift. Needs an ADR before anyone upgrades.
5. **M3-T15** — an evaluation harness (precision/recall/localisation against phantom ground
   truth). Until it exists, "the detector got better" is not a measurable claim.
6. Refresh `README.md` and `project.md`, which still carry pre-M2 claims (M9-T01).
