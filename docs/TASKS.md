# TASKS

**Updated:** 2026-08-04 · **Active:** `M1-T07`

Full task breakdown per milestone. One task ≈ one branch ≈ one focused work session.
Statuses: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` dropped.

Task IDs are permanent. A dropped task keeps its ID; IDs are never reused.
`D-NN` references are defects from `docs/audit/2026-07-28-baseline-audit.md`.

---

## M0 — Engineering foundation ✅

| ID | Task | Status |
|---|---|---|
| M0-T01 | Analyze the existing repository; record strengths and weaknesses with evidence | [x] |
| M0-T02 | Define the target architecture and layer contracts (`Architecture.md`) | [x] |
| M0-T03 | Write the project constitution (`PROJECT_RULES.md`) | [x] |
| M0-T04 | Write initial ADRs (0001–0011) | [x] |
| M0-T05 | Break the project into milestones (`Roadmap.md`) | [x] |
| M0-T06 | Break milestones into tasks (this file) | [x] |
| M0-T07 | Establish the state/session protocol (`STATE.md`, `Progress.md`, `CURRENT_TASK.md`) | [x] |
| M0-T08 | Select the first task and record it | [x] |

---

## M1 — Repository hygiene & quality gates

| ID | Task | Detail | Status |
|---|---|---|---|
| M1-T01 | Untrack build artifacts and weights; rewrite `.gitignore` | Done 2026-08-03: 2 877 → **77** tracked files; 2 800 `node_modules` untracked; 26 MB `yolov8s-world.pt` removed from the index before it entered history; junk deleted; `plan.md` archived; `.claude/settings.json` shared. Zero golden drift. D-19 | [x] |
| M1-T02 | Declare dev dependencies | Done 2026-08-03: pytest 9.1.1, pytest-cov 7.1.0, ruff 0.16.1, mypy 2.3.0. Baseline measured — 196 ruff findings (109 in `src/`), 30 mypy errors, 1 failing test. No runtime version moved; golden stable. D-20 | [x] |
| M1-T03 | Repair the ruff configuration | Done 2026-08-03: removed `fix = true` (`ruff check` was rewriting sources), moved `select`/`ignore` to `[tool.ruff.lint]`, py311 → py312, fixed `known-first-party`, dropped dead `S101`, excluded notebooks. `src/` findings unchanged at 109 — repair only | [x] |
| M1-T04 | Add mypy configuration | Done 2026-08-04: strict for `nanoscope.*` from its first line; `src/` checked but not silenced (22 errors, 13 of them static confirmations of D-01/D-02/D-07/D-10/D-16); `ignore_missing_imports` per module. Three new defects filed as M3-T17…T19 | [x] |
| M1-T05 | Wire the characterization harness into pytest | Done 2026-08-04: `tests/characterization/test_golden.py` calls the existing runner through a new `diff_against_golden()` seam; marked `slow` (192 s), marker registered. `pytest.ini` folded into `pyproject.toml` and deleted. Failure proven by perturbing a golden value | [x] |
| M1-T06 | Replace `tests/test_io.py` | Done 2026-08-04: deleted; replaced by `tests/unit/test_afm_io.py` — 22 tests over a synthetic Nanoscope byte stream (round trip, calibration, unit conversion, 8 failure modes, npy, SEM/TEM). No binary fixture, no `data/`. **`pytest` is green.** Suite validated by killing 4 mutants of the parser; one new defect found and filed as M3-T20. D-20 | [x] |
| M1-T07 | Add pre-commit | ruff, ruff-format, end-of-file-fixer, `check-added-large-files` (1 MB), nbstripout | [ ] |
| M1-T08 | Add CI | GitHub Actions, Python 3.12, CPU-only, no weights, no network: lint → types → tests → golden | [ ] |
| M1-T09 | Clean notebooks | Strip outputs from the committed 6.5 MB and 2.2 MB notebooks; move all to `notebooks/`; mark experimental | [ ] |
| M1-T10 | Add a one-command gate | `Makefile` (or `justfile`) with `check`, `lint`, `types`, `test`, `golden` | [ ] |
| M1-T11 | Decide the fate of `.zip`, `output/`, `__pycache__` at repo root | Absorbed into M1-T01: all deleted and ignored | [x] |

---

## M2 — Domain extraction (behaviour-preserving)

> **Gate for every task in this milestone:** `python tests/characterization/capture.py`
> must report zero drift. Any drift is a bug in the move.

| ID | Task | Detail | Status |
|---|---|---|---|
| M2-T01 | Confirm the package name and create the skeleton | ADR-0011. `nanoscope/{app,core,application,infrastructure,gui,resources}` with `py.typed`; `src/` kept as a shim | [ ] |
| M2-T02 | Extract entities and value objects | `types.py` → `core/entities/` + `core/values/`; add `Modality`, `PixelScale`, `Polarity`, `DeviceKind` | [ ] |
| M2-T03 | Move preprocessing | `preprocess.py` → `core/science/preprocessing/`, unchanged | [ ] |
| M2-T04 | Move I/O parsing | `afm_io.py` → `core/science/io/` (pure parsing) + an `ImageLoader` port implemented in `infrastructure/storage/` | [ ] |
| M2-T05 | Move the LoG detector | `detection/log_detector.py` → `core/science/detection/log.py`; keep `detect()` byte-identical in this commit | [ ] |
| M2-T06 | Move and split measurement | `measure.py` → `core/science/measurement/`; separate modality-neutral geometry from AFM height. D-issue: geometry is trapped in an AFM module | [ ] |
| M2-T07 | Move model-backed code to infrastructure | YOLO and SAM2 wrappers → `infrastructure/models/`; they import torch, so they are not domain | [ ] |
| M2-T08 | Define the ports | `Detector`, `Segmenter`, `ImageLoader`, `ProjectRepository`, `TrainingProvider`, `DeviceProvider`, `LogSink` | [ ] |
| M2-T09 | Break the import cycles + add the import-graph test | Five cycles via `src/__init__.py`; `import src.types` currently loads 1179 modules in 0.67 s. D-18 | [ ] |
| M2-T10 | One owned capability matrix, validated before inference | Rules currently duplicated in `pipeline.py`, `ConfigPanel.tsx` and prose, and already disagreeing. Fixes D-14 | [ ] |
| M2-T11 | Structured logging | Replace 13 `print` calls in library code; `LogSink` port. D-23 | [ ] |
| M2-T12 | English-only library code | 197 Russian lines across five modules; nine reach runtime output. D-22 | [ ] |
| M2-T13 | Retire dead code | 10 unreachable functions; `preprocess_batch.py` fails on every file (D-02) — port to a CLI entry point or delete (operator decision) | [ ] |
| M2-T14 | Package installation | Editable install; delete the `pytest.ini` `pythonpath` hack | [ ] |
| M2-T15 | Delete the `src/` shim | Only when nothing — including notebooks — imports it | [ ] |
| M2-T16 | Refresh `PROJECT_CONTEXT.md` to the new layout | It is the machine-readable map; it must not drift | [ ] |

---

## M3 — Numerical correctness

> **Gate for every task in this milestone:** its own commit, its own ADR, its own golden
> update, and a quantified before/after delta in `Progress.md`. Never bundled.

| ID | Task | Defect | Severity | Status |
|---|---|---|---|---|
| M3-T01 | Fix `build_substrate_map(manual_radius_px=...)` — `UnboundLocalError` on 100% of calls | D-01 | critical | [ ] |
| M3-T02 | Fix `min_size_pixel` flooring to zero on 90% of real scans | D-04 | critical | [!] needs operator decision |
| M3-T03 | Fix YOLO input: normalise **then** cast (12.6% of dynamic range currently survives) | D-03 | critical | [ ] |
| M3-T04 | Aspect-ratio-preserving YOLO letterbox; isotropic box rescale | D-21 | medium | [ ] |
| M3-T05 | Propagate YOLO confidence into `Detection` | D-09 | medium | [ ] |
| M3-T06 | Otsu sizing: raise on empty-after-filter; report post-filter `n_objects` | D-05, D-06 | high | [ ] |
| M3-T07 | Guard LoG normalisation against a zero maximum | D-11 | medium | [ ] |
| M3-T08 | `flatten_lines` must promote dtype like `flatten_plane` does | D-13 | medium | [ ] |
| M3-T09 | Define and apply the opening-radius rounding rule (half-integer radii break `disk()` centring) | D-10 | medium | [!] needs operator decision |
| M3-T10 | Detection polarity: TEM currently returns 0 of 22 particles | D-12 | high | [!] needs operator decision |
| M3-T11 | Handle unknown pixel scale (`None`) without crashing both detectors | D-07 | high | [ ] |
| M3-T12 | Empty measurements must return a schema-stable DataFrame | D-08 | high | [ ] |
| M3-T13 | Typed error taxonomy + input validation at every numerical entry point | D-15 | medium | [ ] |
| M3-T14 | One measurement schema across all four producers; fix the `bbox` contract | D-16, D-17 | medium | [ ] |
| M3-T15 | Evaluation harness: precision / recall / localisation vs phantom ground truth | — | — | [ ] |
| M3-T16 | Characterize `_read_nanoscope_z` against multiple Nanoscope versions | gap | — | [ ] |
| M3-T17 | `_read_nanoscope_z` divides `None` by `samps` when the header has no `Scan Size` — the fallback branch crashes on the next line (`afm_io.py:95-98`) | **new**, found by mypy in M1-T04 | high | [ ] |
| M3-T18 | `YoloDetector._last_result` is initialised to `None`, so its type is `None`; `.filtered_boxes` is accessed unguarded (`yolo_detector.py:50,87,99`) | **new**, mypy | medium | [ ] |
| M3-T19 | `estimate_log_threshold_adaptive` rebinds `responses` from `list[float]` to ndarray before calling `.min()`/`.max()` (`log_detector.py:111,116`) | **new**, mypy | low | [ ] |
| M3-T20 | `load_afm(fmt="npy")` fabricates a physical scale: `pixel_size_nm or 1.0` and `scan_size_nm or float(z.shape[0])` (`afm_io.py:132-133`). Unknown scale must be `None` — the invariant D-07 states. Two consequences: every downstream `_nm` becomes a pixel count wearing nanometre units, and `or` also swallows an explicit `0.0`. Row count is used as a length in nm, which is not even dimensionally a size | **new**, found by the M1-T06 tests | high | [ ] |

---

## M4 — Application layer

| ID | Task | Status |
|---|---|---|
| M4-T01 | Specify the project directory format (layout, schema version, compatibility rules) | [ ] |
| M4-T02 | SQLite schema v1 + a forward-migration mechanism | [ ] |
| M4-T03 | `ProjectRepository` implementation + integration tests | [ ] |
| M4-T04 | Use cases: `CreateProject`, `OpenProject`, `CloseProject`, `ImportImages`, `ListImages` | [ ] |
| M4-T05 | Use cases: `RunDetection`, `RunSegmentation`, `MeasureParticles` | [ ] |
| M4-T06 | Job abstraction: submit, progress, cancel, failure reporting | [ ] |
| M4-T07 | Annotation entity + persistence | [ ] |
| M4-T08 | Undo/redo command stack | [ ] |
| M4-T09 | Autosave service | [ ] |
| M4-T10 | Settings service (application scope + project scope) | [ ] |
| M4-T11 | CSV export service | [ ] |
| M4-T12 | `DeviceManager`: detect CPU / CUDA / ROCm / MPS, apply the selection policy | [ ] |
| M4-T13 | Model registry + `ModelDescriptor` persistence (replaces hardcoded weight paths) | [ ] |
| M4-T14 | Logging infrastructure: structured logs → file + SQLite, with rotation | [ ] |
| M4-T15 | Headless end-to-end integration test of the whole layer | [ ] |

---

## M5 — GUI shell

| ID | Task | Status |
|---|---|---|
| M5-T01 | Entry point, composition root, `nanoscope` console script | [ ] |
| M5-T02 | Main window: menus, toolbars, dockable panels, status bar, layout persistence | [ ] |
| M5-T03 | Dark theme: design tokens + QSS, single source of colour truth | [ ] |
| M5-T04 | Project explorer panel | [ ] |
| M5-T05 | Image viewer: zoom, pan, colormap, LUT range, scale bar, nm coordinate readout | [ ] |
| M5-T06 | ViewModel layer + signal/slot contracts | [ ] |
| M5-T07 | Background job runner with progress and cancellation | [ ] |
| M5-T08 | Log panel + user notifications | [ ] |
| M5-T09 | Settings dialog | [ ] |
| M5-T10 | Headless GUI smoke tests (pytest-qt, offscreen) | [ ] |
| M5-T11 | Architecture lint rule: no `core.science` / `infrastructure` imports in `gui/` | [ ] |

---

## M6 — Analysis workflow in the GUI

| ID | Task | Status |
|---|---|---|
| M6-T01 | Preprocessing panel (flattening, substrate, live preview) | [ ] |
| M6-T02 | Detection panel driven by the capability matrix | [ ] |
| M6-T03 | Detection overlay rendering on the canvas | [ ] |
| M6-T04 | Segmentation panel + mask overlay | [ ] |
| M6-T05 | Measurements table with two-way canvas selection | [ ] |
| M6-T06 | Statistics panel + histograms | [ ] |
| M6-T07 | CSV export UI | [ ] |
| M6-T08 | Multi-image navigation within a project | [ ] |
| M6-T09 | Result persistence and restoration across restart | [ ] |

---

## M7 — Annotation & metrology tools

| ID | Task | Status |
|---|---|---|
| M7-T01 | Annotation layer model + rendering pipeline | [ ] |
| M7-T02 | Point and box tools | [ ] |
| M7-T03 | Polygon tool | [ ] |
| M7-T04 | Brush tool (mask painting) | [ ] |
| M7-T05 | Measurement line + distance tool | [ ] |
| M7-T06 | Profile line + height-profile plot | [ ] |
| M7-T07 | Manual add/edit/delete of detections | [ ] |
| M7-T08 | Undo/redo wired through every tool | [ ] |
| M7-T09 | Annotation export/import in a training-ready format | [ ] |
| M7-T10 | Measurement semantics documented: height, diameter, distance, aspect ratio | [ ] |

---

## M8 — Training module

| ID | Task | Status |
|---|---|---|
| M8-T01 | `TrainingProvider` port: job contract, status, metrics, artifacts | [ ] |
| M8-T02 | Dataset builder: annotations → YOLO dataset with a train/val split | [ ] |
| M8-T03 | `LocalTrainingProvider` (ultralytics), device-aware via `DeviceManager` | [ ] |
| M8-T04 | Training-run persistence: config, metrics, artifacts, provenance | [ ] |
| M8-T05 | Training UI: configuration, live metrics, cancellation | [ ] |
| M8-T06 | Model management UI: import, register, activate, compare | [ ] |
| M8-T07 | `RemoteTrainingProvider`: protocol, client, contract tests | [ ] |
| M8-T08 | Model evaluation report using the M3-T15 harness | [ ] |

---

## M9 — Release readiness

| ID | Task | Status |
|---|---|---|
| M9-T01 | Rewrite `README.md` as a product README; archive stale documents | [ ] |
| M9-T02 | User manual in `docs/` | [ ] |
| M9-T03 | Linux packaging (AppImage or equivalent) + desktop entry | [ ] |
| M9-T04 | Diagnostics bundle for bug reports | [ ] |
| M9-T05 | Performance pass on large scans | [ ] |
| M9-T06 | v1.0 release checklist and tag | [ ] |

---

## Counts

| Milestone | Tasks | Done | Blocked |
|---|---:|---:|---:|
| M0 | 8 | 8 | 0 |
| M1 | 11 | 4 | 0 |
| M2 | 16 | 0 | 0 |
| M3 | 19 | 0 | 3 |
| M4 | 15 | 0 | 0 |
| M5 | 11 | 0 | 0 |
| M6 | 9 | 0 | 0 |
| M7 | 10 | 0 | 0 |
| M8 | 8 | 0 | 0 |
| M9 | 6 | 0 | 0 |
| **Total** | **113** | **12** | **3** |
