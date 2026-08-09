# Roadmap

**Updated:** 2026-08-03 · **Current milestone:** M1

Milestones are ordered by dependency, not by date. Each has a single goal, explicit
exit criteria, and a defined risk to scientific output. Task-level detail lives in
`docs/TASKS.md`.

**Principle:** structure first, correctness second, application third, features fourth.
The scientific core is moved before it is fixed, and fixed before it is wrapped in a UI —
so that every change has a safety net beneath it.

---

## Overview

| # | Milestone | Goal | Moves numbers? | Depends on |
|---|---|---|---|---|
| **M0** | Engineering foundation | Documentation, rules, task system | no | — |
| **M1** | Hygiene & quality gates | A repository that can be worked in | no | M0 |
| **M2** | Domain extraction | Clean Architecture skeleton, science moved verbatim | **no — enforced** | M1 |
| **M3** | Numerical correctness | Fix the 24 audited defects, deliberately | **yes — intentionally** | M2 |
| **M4** | Application layer | Projects, persistence, jobs, devices, models | no | M2 |
| **M5** | GUI shell | Qt6 application that opens, shows, and navigates | no | M4 |
| **M6** | Analysis workflow | Detect / segment / measure / export from the UI | no | M5 |
| **M7** | Annotation & metrology | Manual tools, undo/redo, height profiles | no | M6 |
| **M8** | Training module | Annotations → dataset → model → registry | no | M7 |
| **M9** | Release readiness | Packaging, docs, v1.0 | no | M8 |

---

## M0 — Engineering foundation

**Goal.** Establish how this project is built before building it.

**Scope.** Repository analysis, target architecture, project rules, milestone and task
breakdown, initial ADRs, session/state protocol.

**Exit criteria**
- [x] `docs/PROJECT_RULES.md`, `Architecture.md`, `Roadmap.md`, `TASKS.md`, `Backlog.md`,
      `Progress.md`, `STATE.md`, `CURRENT_TASK.md`, `Development.md` exist and are filled
- [x] ADR-0001 … ADR-0011 written
- [x] Strengths and weaknesses of the current architecture recorded with evidence
- [x] First task selected and written into `CURRENT_TASK.md`

**Status:** ✅ complete (2026-08-03)

---

## M1 — Repository hygiene & quality gates

**Goal.** Make the repository a place where a change can be reviewed and verified.

**Why first.** 98% of tracked files are `node_modules`. No linter, no type checker and
no test runner is installed. Until both are fixed, every subsequent diff is unreviewable
and every "it works" is unverified.

**Scope.** Untrack `node_modules` and model weights, rewrite `.gitignore`, declare dev
dependencies, repair the ruff configuration, add mypy, wire the characterization harness
into pytest, replace the fake test, add pre-commit and CI, strip notebook outputs.

**Exit criteria** — closed 2026-08-04, see the M1 summary in `docs/Progress.md`
- [x] `git ls-files | wc -l` < 100 — **64**
- [x] `ruff check .`, `ruff format --check .`, `mypy`, `pytest` all runnable via one
      command — `make check`, with `mypy` as `make types` outside the blocking gate while
      the legacy core is `src/` (joins `check` in M2-T01)
- [x] `pytest` executes the golden comparison and passes
- [x] CI runs the full gate on every push, CPU-only, no weights, no network
- [ ] No file over 1 MB is tracked — **two README figures remain** (3.2 MB, 3.0 MB);
      filed as **B-054**, deferred to M9-T01. The pre-commit limit stops new ones

**Risk to scientific output:** none. No functional code is touched.

---

## M2 — Domain extraction

**Goal.** The Clean Architecture skeleton exists, and the scientific core lives in it —
with every golden number unchanged.

**Scope.** Create the `nanoscope` package; move `src/` into `core/science` and
`infrastructure/models`; define entities, value objects and ports; introduce the
`Modality` enum and a single owned capability matrix; validate before inference; break
the five import cycles; replace `print` with structured logging; translate Russian
strings; retire dead code.

**Exit criteria** — closed 2026-08-04, see the M2 summary in `docs/Progress.md`
- [x] `python tests/characterization/capture.py` reports **zero drift** after every move —
      sixteen tasks, and the only golden change in the whole milestone was six non-numeric
      lines declared in M2-T12 (four translated exception messages, two `stdout_lines`)
- [x] Import-graph test passes: `core` imports nothing from `gui`/`application`/`infrastructure`
      — static, over the AST, and proven to fail on a real violation (M2-T09)
- [x] `import nanoscope.core.entities` loads no torch, no matplotlib, no Qt — **asserted
      by name** in `tests/unit/test_import_graph.py`, plus no ultralytics, sam2,
      patched_yolo_infer, cv2 or pandas.
      **The "< 100 modules" half of this criterion was wrong and is replaced with 250**
      (M2-T09): numpy alone is 141 modules and the domain is explicitly allowed to use it
      (`Architecture.md` §3), so no module holding an `np.ndarray` annotation can ever
      reach 100. Measured after the fix: **185**, down from 626.
- [~] Ports: **one defined, `Detector`, satisfied by both detectors from opposite layers.**
      Two of the seven planned were removed rather than deferred — `LogSink` because the
      standard library already provides the extension point (**ADR-0013**) — and the rest
      ship with their first adapter, each with a named task in `core/ports/__init__.py`.
      Behaviour is **not** yet reachable through a port: `use_cases/pipeline.py` still
      constructs `YoloDetector` by name. **M4 owns that**, and mypy flags it today
- [x] Zero `print` calls (M2-T11, asserted per module) and zero non-English strings
      (M2-T12) in library code
- [x] `src/` deleted entirely, not just the shims; `pythonpath` deleted outright (M2-T15)

**Risk to scientific output:** must be zero. Any drift is a bug in the refactor.

---

## M3 — Numerical correctness

**Goal.** The pipeline computes what it claims to compute.

**Why after M2.** These changes *will* move golden numbers. They are only safe once the
code is structured, tested and typed, so that each defect can be fixed in isolation with
a measurable delta.

**Scope.** The 24 defects from the Phase 0 audit, prioritised: manual-radius crash (D-01),
`min_size_pixel` floor (D-04), YOLO input corruption (D-03, D-21), Otsu edge cases
(D-05, D-06), degenerate inputs (D-11, D-13, D-15), TEM polarity (D-12), unknown scale
(D-07), empty results (D-08), unified measurement schema (D-16, D-17). Plus an evaluation
harness that scores detection against phantom ground truth.

**Exit criteria**
- [x] Every critical and high defect closed, each with its own commit + ADR + golden update —
      **2026-08-06.** Critical: D-01 (M3-T01), D-02 (ADR-0012, deleted), D-03 (M3-T03),
      D-04 (M3-T02), D-19 (M1-T01). High: D-05/D-06 (M3-T06), D-07 (M3-T11, T20, T17 — three
      faces), D-08 (M3-T12), D-12 (M3-T10), D-18 (M2-T09). What remains in M3 is `medium`
      and below
- [x] Degenerate-input contract documented and tested for every numerical entry point —
      **2026-08-07** (M3-T13, ADR-0030): seven error classes, one `ensure_height_map` at
      **fourteen** entry points, 7 bad inputs x 10 entry points proven to give one error type
- [x] One measurement schema across all four producers — **2026-08-07** (M3-T14, ADR-0031):
      a core every producer emits plus blocks present-in-full or absent-in-full, `method`
      naming the producer; three faults found where the audit named one
- [x] Evaluation harness reports precision/recall/localisation per phantom — **2026-08-07**
      (M3-T15, ADR-0032): `core/science/evaluation.py`, one-to-one optimal assignment, a
      scale-free match radius; **all five criteria are now met**
- [x] Operator has signed off on D-04 semantics (**B2**, ADR-0024) and D-12 polarity
      (**B3**, ADR-0023) — answered 2026-08-05, both executed

**Risk to scientific output:** intentional and quantified. One defect per commit, never
bundled with restructuring.

---

## M4 — Application layer

**Goal.** The concepts the application needs — projects, images, jobs, devices, models,
settings, logs — exist and are persisted, with no UI attached.

**Scope.** Project directory format and SQLite schema v1 with migrations; repositories;
use cases for project/image/analysis lifecycle; the job abstraction; the undo/redo
command stack; autosave; settings; CSV export; structured logging sinks; `DeviceManager`;
the model registry and `ModelDescriptor`.

**Exit criteria**
- [ ] A project can be created, opened, populated with images and closed — from Python, headless
- [ ] Detection and measurement results round-trip through SQLite and the filesystem
- [ ] `DeviceManager` correctly reports CPU / CUDA / ROCm / MPS on this machine and selects one
- [ ] Model registry resolves `"yolo"` and `"sam2"` to providers via `ModelDescriptor`
- [ ] Undo/redo proven on at least one mutating use case
- [ ] Integration tests cover the whole layer; no Qt imported anywhere

**Risk to scientific output:** none. The domain is called, not modified.

---

## M5 — GUI shell

**Goal.** A Qt6 application that starts, opens a project, and shows a scan — with
dockable panels, a dark theme, and no business logic in a single widget.

**Scope.** Entry point and composition root; main window with menus, toolbars, docks and
status bar; dark QSS theme and design tokens; project explorer; image viewer with
zoom/pan, colormap, LUT range and scale bar; the viewmodel layer; the background job
runner with progress and cancellation; log panel; settings dialog; headless GUI tests.

**Exit criteria**
- [ ] `nanoscope` launches on Linux and opens a project created in M4
- [ ] A scan renders with correct nm axes and a scale bar
- [ ] A long-running job shows progress and can be cancelled without freezing the UI
- [ ] GUI smoke tests pass headless in CI
- [ ] Lint rule proves no `gui/` module imports `core.science` or `infrastructure`

**Risk to scientific output:** none.

---

## M6 — Analysis workflow in the GUI

**Goal.** The full existing pipeline is driveable from the application.

**Scope.** Detection panel with LoG and YOLO parameters; detection overlay rendering;
segmentation panel and mask overlay; measurements table linked to the canvas selection;
statistics panel with histograms; CSV export UI; multi-image navigation within a project.

**Exit criteria**
- [ ] Load → detect → segment → measure → export CSV, entirely through the UI
- [ ] Selecting a table row highlights the particle, and vice versa
- [ ] Invalid combinations are disabled in the UI *because* the capability matrix says so — not by a duplicated rule
- [ ] Results persist across application restart

**Risk to scientific output:** none — the UI must not introduce its own defaults.

---

## M7 — Annotation & metrology tools

**Goal.** The operator can correct the machine and measure by hand, GWYDDION-style.

**Scope.** Annotation layer model and rendering; point, box, polygon and brush tools;
measurement line and distance tool; profile line with a height-profile plot; manual
add/edit/delete of detections; undo/redo wired through every tool; annotation
export/import in a training-ready format.

**Exit criteria**
- [ ] All seven annotation/measurement tools usable and persisted
- [ ] Undo/redo covers every mutating action, including brush strokes
- [ ] A height profile along a drawn line matches the notebook implementation
- [ ] Annotations export to a format the M8 dataset builder consumes

**Risk to scientific output:** none for automatic paths; manual measurements are a new
output and get their own tests.

---

## M8 — Training module

**Goal.** The operator's own annotations become a model, inside the application.

**Scope.** `TrainingProvider` port with job/status contract; dataset builder from
annotations; `LocalTrainingProvider` on ultralytics; training-run persistence and
metrics; training UI with live metrics and cancellation; model management UI (import,
register, activate, compare); `RemoteTrainingProvider` protocol and client.

**Exit criteria**
- [ ] Annotations → dataset → trained weights → registered `ModelDescriptor`, without leaving the app
- [ ] Training runs as a cancellable job; metrics stream to the UI and persist
- [ ] A trained model is selectable for detection in M6 with no code change
- [ ] `RemoteTrainingProvider` satisfies the same port and is covered by contract tests

**Risk to scientific output:** new models change detections by design. Model comparison
is reported through the M3 evaluation harness.

---

## M9 — Release readiness

**Goal.** Someone who is not the author can install and use it.

**Scope.** Rewrite `README.md` as a product README; archive stale documents; user
manual; Linux packaging (AppImage or equivalent) with a desktop entry; diagnostics
bundle for bug reports; performance pass on large scans; v1.0 checklist.

**Exit criteria**
- [ ] Clean-machine install and first-run succeed from the documented steps alone
- [ ] `README.md` contains no claim contradicted by the code
- [ ] Large-scan performance measured and documented
- [ ] v1.0 tagged

---

## Sequencing rules

1. **M2 before M3.** Never fix numerics inside a moving structure.
2. **M4 before M5.** The GUI is written against a layer that already works headless.
3. **M3 can run in parallel with M4** — different files, different risk profiles — but a
   golden update and a use case never share a commit.
4. **M7 before M8.** Training needs annotations to exist first.
5. Any milestone may spawn ADRs. No milestone may skip its exit criteria.
