# Architecture

**Status:** target architecture, not yet implemented · **Version:** 1.0 · **Updated:** 2026-08-03

This document describes where the project is going. For what the code does *today*,
read `PROJECT_CONTEXT.md` and `docs/audit/2026-07-28-baseline-audit.md`.

---

## 1. What we are building

A Linux desktop application for nanoparticle microscopy analysis: load a scan, detect
particles, segment them, measure them, annotate them by hand, collect statistics,
export results, and train detection models on your own annotations — all inside one
offline application, on the operator's own machine.

The scientific pipeline already exists and works. The application does not exist. This
architecture is about wrapping proven science in an application that can be maintained
for years.

---

## 2. Current state — analysis

### 2.1 What is there today

```
src/                          2021 LOC, 12 modules — the working scientific core
├── types.py                  dataclasses (Detection, PipelineConfig, PipelineResult, …)
├── afm_io.py                 custom Bruker Nanoscope SPM parser + SEM/TEM image loader
├── preprocess.py             plane/line flattening, morphological substrate estimation
├── preprocessing_pipeline.py load + preprocess orchestration
├── detection/                BaseDetector ABC, LoG detector, YOLOv8 detector
├── segmentation.py           SAM2 prompting, mask selection, measurement
├── measure.py                circular baseline height measurement, mask geometry
├── pipeline.py               detector × mode dispatcher
└── visualization.py          matplotlib plots and an interactive viewer

frontend/                     React + TS + Vite client for a backend that was never written
notebooks/, *.ipynb           experiments, some committed with outputs
tests/                        1 non-test + a Phase 0 characterization harness (good)
docs/audit/                   Phase 0 audit: 24 confirmed defects, golden baseline
```

### 2.2 Strengths — what we keep

| Strength | Why it matters |
|---|---|
| **The science works.** Flatten → substrate → detect → measure is a coherent, physically motivated chain, accurate on clean fields (24/24, 30/30, 14/14 particles on the phantom set). | This is the asset. Everything else is replaceable. |
| **A modality-neutral `Detection` type and a `BaseDetector` ABC already exist.** | The pluggable-provider requirement is half-built already. |
| **SAM2 is isolated behind lazy imports.** | Segmentation can be swapped or omitted without touching detection. |
| **`types.py` was written as a deliberate dependency root.** | The intent to layer the code predates this document. |
| **Dataclass-based contracts** (`PreprocessingResult`, `PipelineResult`) rather than loose tuples/dicts. | Gives the application layer something typed to build on. |
| **A characterization harness with seeded phantoms and a committed golden file already exists.** | We can restructure aggressively without gambling on the numerics. This is rare and valuable. |
| **A completed, reproduced Phase 0 audit.** | We start with a measured defect register instead of guesses. |
| **Two detector backends and a tiled-inference path.** | The classical/learned comparison the science needs is already wired. |

### 2.3 Weaknesses — what must change

| # | Weakness | Evidence | Fixed in |
|---|---|---|---|
| W1 | **No application layer at all.** The code is a library plus notebooks. No projects, no persistence, no jobs, no settings, no logging. | `src/` has no state management of any kind | M4 |
| W2 | **No UI.** The only client is a React app talking to an HTTP backend that does not exist. | `PROJECT_CONTEXT.md` §4, §11 | M5–M7 (Qt), ADR-0007 |
| W3 | **Five import cycles; the "dependency root" pulls in 1179 modules.** | Audit D-18: `import src.types` → 0.67 s, matplotlib + pandas loaded | M2 |
| W4 | **Modality is inferred from types, not modelled.** `isinstance(data, PreprocessingResult)` decides whether input is AFM. | `pipeline.py:44`, `segmentation.py:70` | M2-T10 |
| W5 | **The capability matrix lives in three places and already disagrees with itself.** | `pipeline.py:88`, `ConfigPanel.tsx:109`, `PROJECT_CONTEXT.md` §9 | M2-T10 |
| W6 | **Validation runs after inference.** AFM+YOLO+baseline burns a full inference pass, then raises. | Audit D-14 | M2-T10 |
| W7 | **Silent numerical defects on real data.** YOLO input keeps 12.6% of dynamic range; the minimum-size filter is off on 90% of the operator's scans. | Audit D-03, D-04, measured | M3 |
| W8 | **No device management.** Nothing decides CPU vs CUDA vs ROCm vs MPS; it is implicit in torch defaults. | no such module exists | M4-T12 |
| W9 | **No training path.** Models are consumed, never produced. Annotations cannot become a dataset. | no such module exists | M8 |
| W10 | **Model paths are hardcoded config strings** (`"./checkpoints/best12x.pt"`). No registry, no metadata, no versioning. | `types.py:79` | M4-T13 |
| W11 | **No error taxonomy.** Ten malformed inputs produce five different exception types, none naming the parameter. | Audit D-15, characterization §3.5 | M3-T13 |
| W12 | **Four measurement producers emit four different schemas.** The same SAM2 score ships as `score` and as `sam_score`. | Audit D-17 | M3-T14 |
| W13 | **Repository hygiene is broken.** 2 800 tracked `node_modules` files = 98% of the repo; a 137 MB checkpoint is staged into history. | measured, audit D-19 | M1-T01 |
| W14 | **No quality gate.** pytest, ruff and mypy are not declared or installed; the only test has no assertions. | audit D-20 | M1 |
| W15 | **`print`-based diagnostics, partly in Russian, reaching users.** | audit D-22, D-23 | M2-T11, M2-T12 |
| W16 | **Dead code on the main paths**: 10 unreachable functions, and `preprocess_batch.py` fails on every file. | audit D-02, §1 | M2-T13 |

**Summary judgment.** The domain layer is sound and worth preserving almost verbatim.
Everything above it is missing, and everything around it (packaging, gates, hygiene) is
absent. The right move is not a rewrite — it is extraction: lift the science into a
clean `core`, build the missing layers on top, and fix the numerics separately under
the protection of the existing golden file.

---

## 3. Target architecture

Clean Architecture, four rings, one composition root.

```
┌───────────────────────────────────────────────────────────────────────┐
│  app/           composition root — bootstrap, DI wiring, entry point   │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ constructs everything
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
┌───────────────┐   ┌───────────────────────┐   ┌────────────────────────┐
│  gui/         │──▶│  application/         │◀──│  infrastructure/       │
│  PySide6      │   │  use cases, services  │   │  adapters              │
│  views        │   │  DTOs, capabilities   │   │  sqlite, fs, models,   │
│  viewmodels   │   │  job orchestration    │   │  device, training, log │
└───────────────┘   └───────────┬───────────┘   └───────────┬────────────┘
                                │                           │
                                ▼                           │ implements
                    ┌───────────────────────────────────────▼────────────┐
                    │  core/                                             │
                    │  entities · value objects · ports · science        │
                    │  pure Python + NumPy. No Qt. No torch. No I/O.     │
                    └────────────────────────────────────────────────────┘
```

**The dependency rule:** arrows point inward. `core` knows nothing about anything else.

### 3.1 Package layout

```
nanoscope/                          # ADR-0011 — name pending operator confirmation
├── app/
│   ├── __main__.py                 # entry point: `nanoscope`
│   ├── bootstrap.py                # builds the container, wires ports → adapters
│   └── container.py                # explicit dependency container (no magic DI)
│
├── core/                           # DOMAIN — the preserved scientific core
│   ├── entities/                   # Project, ImageRecord, Detection, Measurement,
│   │                               #   Annotation, TrainingRun, ModelDescriptor
│   ├── values/                     # Modality, PixelScale, Units, Polarity, DeviceKind
│   ├── errors.py                   # the project error taxonomy
│   ├── ports/                      # abstract interfaces — the contracts
│   │   ├── detector.py             #   Detector
│   │   ├── segmenter.py            #   Segmenter
│   │   ├── image_loader.py         #   ImageLoader
│   │   ├── repositories.py         #   ProjectRepository, MeasurementRepository, …
│   │   ├── training.py             #   TrainingProvider
│   │   ├── device.py               #   DeviceProvider
│   │   └── logging.py              #   LogSink
│   └── science/                    # ← today's src/, moved, not rewritten
│       ├── io/                     #   SPM header parsing, calibration (pure)
│       ├── preprocessing/          #   flatten_plane, flatten_lines, substrate
│       ├── detection/              #   LoG (pure NumPy/skimage)
│       ├── measurement/            #   height baseline, mask geometry
│       └── constants.py            #   SIGMA_TO_RADIUS = sqrt(2), etc.
│
├── application/
│   ├── use_cases/                  # CreateProject, ImportImages, RunDetection,
│   │                               #   RunSegmentation, MeasureParticles, ExportCsv,
│   │                               #   StartTraining, …
│   ├── dto/                        # boundary types: GUI ↔ application
│   ├── capabilities.py             # THE capability matrix (modality × detector × mode)
│   ├── jobs.py                     # job abstraction: submit, progress, cancel
│   ├── commands.py                 # undo/redo command stack
│   └── services/                   # AutosaveService, SettingsService, ExportService
│
├── infrastructure/
│   ├── persistence/                # SQLite schema, migrations, repository impls
│   ├── storage/                    # project directory layout, file naming, cache
│   ├── models/                     # YoloDetectorProvider, Sam2SegmenterProvider,
│   │                               #   registry, weight metadata
│   ├── training/                   # LocalTrainingProvider, RemoteTrainingProvider
│   ├── device/                     # DeviceManager — CPU / CUDA / ROCm / MPS
│   ├── imaging/                    # colormaps, PNG/QImage rendering, overlays
│   └── logging/                    # structured logging → file + SQLite
│
├── gui/                            # PySide6 only. No business logic.
│   ├── main_window.py
│   ├── views/                      # ProjectExplorer, ImageViewer, Measurements,
│   │                               #   Statistics, Training, ModelManager, LogPanel
│   ├── widgets/                    # reusable controls, canvas items, tools
│   ├── viewmodels/                 # per-view state, talks to use cases only
│   ├── tools/                      # annotation tools: point, box, polygon, brush,
│   │                               #   line, profile, distance
│   └── theme/                      # dark QSS, design tokens
│
└── resources/                      # icons, qss, translations, sample data

tests/
├── unit/            core + application, fast, no I/O
├── integration/     infrastructure: sqlite, storage, providers (mocked weights)
├── gui/             pytest-qt, offscreen
├── characterization/ existing golden harness — the refactor safety net
└── evaluation/      detection quality vs phantom ground truth (M3-T15)
```

### 3.2 Layer contracts

| Layer | May import | May **not** import | Owns |
|---|---|---|---|
| `core` | numpy, scipy, skimage, pandas, stdlib | Qt, torch, sqlite3, requests, anything in this project outside `core` | the science, the entities, the ports |
| `application` | `core` | Qt, torch, sqlite3, filesystem | use cases, capability rules, jobs, undo/redo |
| `infrastructure` | `core`, third-party SDKs | `gui`, `application` | adapters that satisfy ports |
| `gui` | `application`, `core.entities`, `core.values`, PySide6 | `core.science`, `infrastructure`, torch | presentation and interaction |
| `app` | everything | — | wiring, and nothing else |

An automated import-graph test enforces this table (M2-T09). It fails the build, not a
code review.

---

## 4. Key mechanisms

### 4.1 Model providers

Detectors and segmenters are ports with pluggable implementations, resolved through a
registry keyed by a string identifier that comes from configuration — never from an
`if/elif` chain in the pipeline.

```python
# core/ports/detector.py
class Detector(Protocol):
    def detect(self, image: NDArray, scale: PixelScale, *, ctx: DetectionContext) -> list[Detection]: ...

# infrastructure/models/registry.py
registry.register("log",  LogDetectorProvider)
registry.register("yolo", YoloDetectorProvider)
```

Adding a model means adding a provider and one registry line. No other file changes.
Model weights are described by a `ModelDescriptor` (id, task, framework, path, input
size, class map, provenance, checksum) stored in the database, not by a raw path string
in a config dataclass. See ADR-0005.

### 4.2 Device manager

One component answers "where does this run?". It probes CUDA, ROCm, MPS and CPU,
applies a selection policy (explicit user choice → best available → CPU), and hands
providers a resolved device. No provider calls `torch.cuda.is_available()`. See ADR-0004.

### 4.3 Training

`TrainingProvider` is a port with two implementations: `LocalTrainingProvider` (spawns
training on this machine) and `RemoteTrainingProvider` (submits to a remote worker).
Training is an **application module**, not a branch inside detection: it consumes
annotations, produces a `ModelDescriptor` and a `TrainingRun` record, and never touches
the detection code path. The GUI talks only to the port. See ADR-0006.

### 4.4 Project storage

A project is a directory the operator owns:

```
MyProject/
├── images/         source scans, untouched
├── annotations/    manual annotations (JSON/mask files)
├── results/        detections, measurements, masks
├── exports/        generated CSV
├── models/         project-local model weights
├── logs/           run logs
├── cache/          derived artifacts — safe to delete
└── database.sqlite metadata only, no binaries
```

See ADR-0003 and M4-T01 for the format specification.

### 4.5 Jobs, undo/redo, autosave

- **Jobs.** Anything that can take longer than ~100 ms (detection, segmentation,
  training, import) runs as a job with progress and cancellation. The GUI subscribes;
  it does not block and it does not own the thread policy.
- **Undo/redo.** A command stack in `application/commands.py`. Every mutating user
  action is a command with `do()` and `undo()`. The GUI dispatches commands; it never
  mutates state directly.
- **Autosave.** A service that persists dirty state on a timer and on well-defined
  events. Enabled by default.

### 4.6 Error taxonomy

One hierarchy in `core/errors.py`: `NanoscopeError` → `ValidationError`,
`UnsupportedFormatError`, `CalibrationError`, `DegenerateInputError`,
`CapabilityError`, `ModelError`, `DeviceError`, `StorageError`. Every error names the
offending parameter and its value. Internal NumPy/SciPy exceptions never reach a user
surface. Replaces the current five-exception-types-for-ten-inputs situation (D-15).

---

## 5. Migration strategy

The move from `src/` to `nanoscope/` is a **pure relocation**, executed under the
protection of the characterization golden file:

1. Create the package skeleton; keep `src/` importable as a thin shim (M2-T01).
2. Move one module at a time, largest blast radius last: entities → preprocessing →
   io → LoG → measurement → YOLO/SAM2 providers.
3. After each move: `python tests/characterization/capture.py` must report **zero drift**.
4. Break the import cycles; add the import-graph test.
5. Delete the shim only when nothing imports `src` — including notebooks.
6. **Only then** start M3, where numbers are allowed to change, one defect per commit,
   each with an ADR.

Segmentation moves last: it has no golden coverage (SAM2 inference is not reproducible
in CI), so it is the least protected module in the repository.

---

## 6. What we are deliberately not building

| Not building | Why | Revisit |
|---|---|---|
| HTTP backend / web client | The product is an offline desktop app. The React client is parked, not deleted. | ADR-0007, Backlog |
| Batch processing | Explicitly out of scope for v1. | Backlog |
| Plugin system | Not needed now — but the registry and port layout make it addable without redesign. | Backlog |
| Cloud storage / multi-user | Single-operator, single-machine, offline-first. | Backlog |
| Non-CSV export formats | CSV only for v1. | Backlog |

---

## 7. Open architectural questions

Tracked in `docs/STATE.md` under *Blocked / needs decision*:

1. **Package name** — `nanoscope` proposed; the distribution is still `afm-analysis`. (ADR-0011, Proposed)
2. **`min_size_nm` semantics** (D-04) — what *should* the minimum particle size mean
   when a pixel is 9.77 nm? Requires the operator, not the engineer.
3. **Detection polarity** (D-12) — explicit configuration, or auto-detected from the
   image? TEM currently detects the background.
4. **Fate of `frontend/`, `preprocess_batch.py`, and the committed notebooks** — park,
   archive, or delete. Deletion needs the operator's approval.
5. **Real sample data in git** — the phantom set stands in for it today; committing a
   real scan is the operator's call.
