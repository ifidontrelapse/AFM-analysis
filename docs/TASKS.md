# TASKS

**Updated:** 2026-08-06 · **Active:** every `critical` and `high` defect is closed; next are `M3-T08`, `M3-T13`, `M3-T14` and `M3-T15`. `M3-T02`, `M3-T20`, `M3-T17`, `M3-T12`, `M3-T05` closed 2026-08-06, `M3-T09`, `M3-T10`, `M3-T21` and `B-058` on 2026-08-05

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
| M1-T07 | Add pre-commit | Done 2026-08-04: 9 hooks, each demonstrated failing on a deliberately bad staged file. ruff runs as a `repo: local` hook on the project's own version, so no second version is ever declared. Rewriting hooks skip `src/` + `preprocess_batch.py` (109 findings there would block every M2 commit); refusing hooks apply everywhere. pytest/mypy stay out of the commit path → CI | [x] |
| M1-T08 | Add CI | Done 2026-08-04, **green after four runs**. Workflow written and green locally, but the first real runs on GitHub found what local verification could not. Run 1: red at pytest, no readable reason (job logs need admin) → diagnostics added, failures now go to annotations + run summary. Run 2: red on my error — `setup-uv` publishes no floating `@v9` tag; both actions now pinned exactly. Run 3: red on **one** golden difference, an exception *message*, not a number: CI resolved Python 3.14 (`requires-python = ">=3.12"`) and 3.14 reworded `too many values to unpack`. Python 3.12 pinned + asserted → **run 4 green**. Underlying fragility filed as **B-058** | [x] |
| M1-T09 | Clean notebooks | Done 2026-08-04: outputs stripped via the configured hook — **8.3 MB → 32 KB**, all 45 cells of code intact; both moved to `notebooks/` with a README; `main.ipynb` (0 bytes, invalid JSON) deleted. Tracked working tree 17 MB → **7.8 MB**. **`pre-commit run --all-files` is green for the first time.** Stale references fixed in `README.md` and `project.md` | [x] |
| M1-T10 | Add a one-command gate | Done 2026-08-04: `Makefile`, 53 lines, targets `check` `format` `lint` `test` `fast` `golden` `types` `lint-legacy`, bare `make` prints the list. **CI calls the targets** — the workflow no longer contains a copy of any command, which is the point of the task. Proven to fail closed: a deliberately misformatted file stopped `check` at step 1 in 0.04 s with exit 2, and a failing test failed the target. `types`/`lint-legacy` stay outside `check` because the legacy baseline is non-zero by design; a gate that can never pass gets bypassed. `pretty = true` dropped from `[tool.mypy]` — the target passes `--no-pretty`, so it was dead config | [x] |
| M1-T11 | Decide the fate of `.zip`, `output/`, `__pycache__` at repo root | Absorbed into M1-T01: all deleted and ignored | [x] |

---

## M2 — Domain extraction (behaviour-preserving)

> **Gate for every task in this milestone:** `python tests/characterization/capture.py`
> must report zero drift. Any drift is a bug in the move.

| ID | Task | Detail | Status |
|---|---|---|---|
| M2-T01 | Create the package skeleton | Done 2026-08-04: `nanoscope/{app,core,application,infrastructure,gui,resources}` + `py.typed`, each layer's `__init__` stating its half of the dependency rule. Distribution `afm-analysis` → `nanoscope`; `uv.lock` regenerated and the diff read — **only the project entry moved, 0 of 119 dependency versions changed**, which matters because CI runs `uv sync --locked` and the golden is version-sensitive. mypy now checks 20 files instead of 13 and the strict `nanoscope.*` override **binds for the first time** (M1-T04 wrote it before the package existed; it had been reported unused ever since). No sub-packages below the layer level — those arrive with the code in M2-T02…T08. Zero code moved, golden zero drift | [x] |
| M2-T02 | Extract entities and value objects | Done 2026-08-04, **three commits so drift would be attributable**. (1) The six dataclasses moved `src/types.py` → `nanoscope/core/entities/`; `src/types.py` is now a shim that defines nothing. Proven mechanically before the gate: identical fields/order/defaults against the pre-move module, and `src.types.X is nanoscope.core.entities.X` — **one `Detection` class, not two**, or `isinstance` lies across the boundary. (2) The strict `nanoscope.*` override caught what legacy code arriving verbatim does not satisfy: bare generics tightened to `dict[str, Any]`; `Detection.bbox` given a scoped `type: ignore` because mypy complaining there **is** D-16, and fixing it moves a number the golden records — `warn_unused_ignores` makes that ignore expire itself when M3 lands. nanoscope back to **0 errors**. (3) `Modality`, `Polarity`, `PixelScale`, `DeviceKind` + 8 mutation-validated tests, **defined but adopted by nothing** — adoption changes `asdict` output, so it belongs to M2-T10 / M3-T10 / M2-T03…T07 / M4-T12. **M2-T13 must not delete them.** `StrEnum` over `(str, Enum)` — ruff UP042 pointed at the stdlib. Golden: zero drift | [x] |
| M2-T03 | Move preprocessing | Done 2026-08-04: → `core/science/preprocessing/`, split into `flatten.py` and `substrate.py`; `src/preprocess.py` is a shim. **Proved by AST comparison before the gate ran** — all six functions code-identical, docstrings differing only in trailing whitespace, and the 5 mypy errors moved with the code (21 before, 21 after). Golden: zero drift. The task that mattered was the collision this exposed: verbatim legacy cannot satisfy a strict package, so the transit status is declared **once in configuration** — mypy at default strictness for `nanoscope.core.science.*`, ruff blocking there but ignoring six named rules (Russian text → M2-T12, `print` → M2-T11, implicit-optional → M3, RET504 cosmetic). Nothing silenced, every entry carries the task that deletes it | [x] |
| M2-T04 | Move I/O parsing | Done 2026-08-04: SPM decoding → `core/science/io/nanoscope_spm.py`; `load_afm`, `load_microscopy_image`, `make_synthetic_afm` → `infrastructure/storage/loaders.py`, because each takes a path and opens it. `src/afm_io.py` is a shim. **The `ImageLoader` port was deliberately not written here** — a Protocol with one implementation and no second caller is what M2-T08 defines wholesale, and writing it twice is writing it twice. Three of ruff's safe fixes landed in `loaders.py` (RET505, UP037, PIE790) and are named rather than hidden behind a claim of verbatim | [x] |
| M2-T05 | Move the LoG detector | Done 2026-08-04: `log_detector.py` and `base.py` → `core/science/detection/`; both shims now. **All 7 definitions AST-identical** — `detect_particles` is the most golden-covered function in the project, recorded for all 8 phantoms at `rtol=1e-6`, and it did not move a number. `yolo_detector.py` stays in `src/` on purpose: it imports torch, so it is infrastructure and belongs to M2-T07 | [x] |
| M2-T06 | Move and split measurement | Done 2026-08-04: → `core/science/measurement/`, split `height.py` (needs a Z map — AFM) from `geometry.py` (needs only a binary mask — any modality). All 5 definitions AST-identical. **This is the split that fixes the D-issue**: mask geometry was trapped in an AFM-named module, which is why `src/segmentation.py` reaches into `src.measure` for SEM/TEM shape metrics; that dependency is now on a modality-neutral module | [x] |
| M2-T07 | Move model-backed code to infrastructure | Done 2026-08-04: `YoloDetector` → `infrastructure/models/yolo.py`, SAM2 runners → `models/sam2.py`, and `afm_to_rgb`/`overlay_masks` → `infrastructure/imaging/colormap.py` (neither has anything to do with SAM2 — same accident M2-T06 untangled). **After this commit nothing under `core` imports torch, ultralytics, sam2 or patched_yolo_infer**, which is what makes the dependency rule true rather than aspirational. AST-verified 4/6 identical, the other two named. `F821` caught two dangling `afm_to_rgb` references before any test ran — the argument for keeping ruff blocking on moved code. mypy 21 → 21, after both modules joined `core.science` at default strictness (strict produced 16 errors in one commit); `infrastructure.storage.loaders` deliberately not exempted, since it passes strict as written | [x] |
| M2-T08 | Define the ports | Done 2026-08-04, **but only `Detector`** — a deliberate narrowing, not an omission. The other six have no implementation, no caller and no second candidate implementation in the repository; an interface written before its first adapter is a guess that gets rewritten once real code has to fit through it, by which point it is quoted in a document and looks decided. `core/ports/__init__.py` carries a table of which task brings each one (LogSink → M2-T11, DeviceProvider → M4-T12, ProjectRepository → M6, TrainingProvider → M7, …); that table is the commitment. `Detector` is a `Protocol` satisfied **today** by `LogDetector` (core) and `YoloDetector` (infrastructure) from opposite layers, neither importing `core.ports`. It does not replace `BaseDetector`, which is inherited and carries `_blobs_to_detections`. 3 tests, the strongest of which is mypy checking the signature structurally; one asserts `import nanoscope.core.science` leaves torch out of `sys.modules` | [x] |
| M2-T09 | Break the import cycles + add the import-graph test | Done 2026-08-04. All five cycles had **one cause** — `src/__init__.py` re-exported the pipeline and detectors, and Python runs a package `__init__` before any submodule, so importing the "dependency root" pulled in SAM2 and matplotlib. Nothing ever used `from src import X`; emptying one file broke all five at no cost to any caller. `import src.types` **1198 → 187 modules, 0.77 s → 0.07 s**; `nanoscope.core.entities` **626 → 185**, matplotlib and pandas gone (the latter moved behind `TYPE_CHECKING` — `from __future__ import annotations` never evaluates it). `tests/unit/test_import_graph.py`: direction checked **statically over the AST** (catches an edge in a module no test runs), weight checked **dynamically in a subprocess**. Both proven to fail; includes a guard against the glob matching nothing. **The M2 exit criterion's "< 100 modules" was unachievable and is corrected to 250** — numpy alone is 141 | [x] |
| M2-T10 | One owned capability matrix, validated before inference | Done 2026-08-04: `nanoscope/application/capabilities.py` is the one executable copy; `PROJECT_CONTEXT.md`'s table now documents it rather than restating it (the third copy, `ConfigPanel.tsx`, was deleted by ADR-0012 — it was the one that had drifted, D-19). **D-14 fixed**: `validate_request` runs before a detector is constructed, so AFM + YOLO + baseline no longer burns a full inference pass before raising. Every rejection message is byte-identical to the old one, in the same most-specific-first order. 12 tests carry the change alone — the golden never calls `run_pipeline` — and the key ones monkeypatch both detectors to explode on construction; commenting out the call turns 2 red | [x] |
| M2-T11 | Structured logging | Done 2026-08-04: 13 `print` calls → `logging.getLogger(__name__)`, lazy `%`-formatting, no library-side configuration. **No `LogSink` port — `ADR-0013`**: it would only wrap `logging`, whose `Handler` is already the extension point, `LogRecord` already the structured payload, and whose handlers are already attached by the application. The SQLite destination becomes a `Handler` in M6. `ADR-0001`'s port list amended. 41 tests: a per-module AST sweep plus real call paths through `caplog`, including silence-by-default. **The golden caught a bug here** — `"1%%"` is only an escape when `logging` formats, which it does not without args | [x] |
| M2-T12 | English-only library code | Done 2026-08-04: 197 lines across **six** modules (`visualization.py` translated in place though it has not moved). `grep -rn "[а-яА-ЯёЁ]"` over `nanoscope/ src/ tests/` returns nothing. **First declared golden change in the project**, and the diff is the argument for having one: **6 lines, not one a number** — 4 translated exception messages plus `stdout_lines` 8→0 and 4→0, the latter being M2-T11 arriving. Re-baselined with `--write`; re-compare clean | [x] |
| M2-T13 | Retire dead code | Done 2026-08-04: **4 deleted, not 10** — `run_full_pipeline`, `plot_pipeline_result`, `plot_detections_histogram`, `make_synthetic_afm`. **Reporting four is the finding.** The audit counted callers; six of its ten are load-bearing for reasons a caller count cannot see: `load_microscopy_image` is the only SEM/TEM entry point and has 4 tests; **`estimate_log_threshold` is recorded by the golden on every phantom** and is the baseline the adaptive variant was adopted against; `run_preprocessing` is the documented entry point M4 wires up; three more are used by the notebooks M1-T09 kept | [x] |
| M2-T14 | Package installation | Done 2026-08-04: `[build-system]` hatchling, `packages = ["nanoscope"]`; `src/` deliberately unpackaged (shipping it would publish `import src`, the collision ADR-0011 renamed away from). Wheel verified: `py.typed` in, no `src/`, 37 modules. **The `pythonpath` hack is half gone**: the `"src"` entry — the real one, which shadowed stdlib `types` and `pipeline` — is deleted; `"."` stays until M2-T15 removes the shims, and the reason is now written down. Also established: **CI does not install the project** (`--only-group ci` installs the group only, deliberately, or torch would return), so CI resolves `nanoscope` through that same entry | [x] |
| M2-T15 | Delete the `src/` shim | Done 2026-08-04: **`src/` deleted entirely.** The title understated it — three modules had never had a shim (`pipeline`, `preprocessing_pipeline`, `visualization`) and had to move first: the two orchestrations to `application/use_cases/`, plotting to `infrastructure/imaging/`. Callers rewired: the golden harness (7 sites), three test modules, both notebooks. **`pythonpath` deleted outright**, mypy now points at one package. A test caught a naming trap — a module and a function with the same name shadow each other through `__init__`, so `use_cases/run_pipeline.py` became `pipeline.py` | [x] |
| M2-T16 | Refresh `PROJECT_CONTEXT.md` to the new layout | Done 2026-08-04: it had drifted past usefulness — describing `src/`, the ADR-0012 frontend, a `pytest.ini` deleted in M1-T05 and a batch script broken since `AFMRawData` landed. Repository map, layer diagram, dependency direction, every path in §5–§10, dependencies, gates, gaps and agent guidance rewritten. The dependency rule is now described as **a test** with the file enforcing it and the measured import weight; the gaps section names the audit ID and M3 task for every defect M2 deliberately did not fix | [x] |

---

## M3 — Numerical correctness

> **Gate for every task in this milestone:** its own commit, its own ADR, its own golden
> update, and a quantified before/after delta in `Progress.md`. Never bundled.

| ID | Task | Defect | Severity | Status |
|---|---|---|---|---|
| M3-T01 | Fix `build_substrate_map(manual_radius_px=...)` — `UnboundLocalError` on 100% of calls | D-01 | critical | [x] Done 2026-08-04, **ADR-0014**. One line: `opening_radius = manual_radius_px`, the value actually passed to the opening. **No rounding, no floor** — both would pre-empt B4/M3-T09 and M3-T13. Delta: **50 golden differences, all under `build_substrate_map_manual`** (10 fields × 5 phantoms); the automatic path 100% of real callers use is untouched. The harness was extended in the same commit to record the returned arrays, or fixing the defect would have left the branch *less* characterized than while broken. 6 tests; restoring the bug turns 5 red |
| M3-T02 | Fix `min_size_pixel` flooring to zero on 90% of real scans | D-04 | critical | [x] Done 2026-08-06, **ADR-0024**, decision **B2 → filter in nanometres, delete the `int()`**. `estimate_radius_otsu` and `estimate_rough_radius` take `min_size_nm`; the comparison is `radii_nm >= min_size_nm` and the three conversion sites in `build_substrate_map` are gone. Delta: **47 golden differences — 27 changed, 15 added, 5 removed**; **`afm_sparse_low_snr` 75 → 17 objects** and everything derived from its radii, the other four AFM phantoms byte-identical, **no measured height moves anywhere** (the final opening radius is 8 on both sides). Re-reading all **628** scan headers reproduces the audit's 90% and adds the part it did not measure: the zero threshold cost nothing on the **365 (58%)** scans coarser than 8.86 nm/px, where one pixel is already over 5 nm; it cost the noise filter on the **203 (32%)** in the 5–8.86 band; and the finest **60 (10%)** were hurt by truncation rather than by the floor, which is `afm_sparse_low_snr`'s case (2.5 px → 2 px). 5 tests; restoring the `int()` turns 3 red. The duplicated `radii_nm` assignment went with it — the change forced the line to move |
| M3-T03 | Fix YOLO input: normalise **then** cast (12.6% of dynamic range currently survives) | D-03 | critical | [x] Done 2026-08-05, **ADR-0015**. The cast moved after the normalisation; `min_size`-style semantics untouched. Delta: **67 golden differences, all under `yolo_input_preparation`**, on all 7 phantoms. Grey levels reaching the network: 8–208 → 239–256. The harness's D-03 measuring stick, `mean_abs_diff_vs_normalize_first`, now reads **0.0** everywhere and becomes a permanent guard. 6 tests; restoring the order turns 5 red. **Not claimed: better detections** — the weights were trained on images the old path produced (see the ADR) |
| M3-T04 | Aspect-ratio-preserving YOLO letterbox; isotropic box rescale | D-21 | medium | [x] Done 2026-08-05, **ADR-0016**. Isotropic scale to fit, pad to the square with 255 (substrate after inversion, applied *after* the normalisation), and one shared geometry helper so the forward and inverse maps cannot drift. Delta: **0 golden differences, 7 keys added** — a square scan is byte-identical, and every phantom is square, which is why the harness gained `non_square_half_height`. 5 geometry tests; restoring the squash turns 4 red. Found while reading: **M3-T21** |
| M3-T05 | Propagate YOLO confidence into `Detection` | D-09 | medium | [x] Done 2026-08-06, **ADR-0028**. The model scores every box, `cfg.yolo_conf` filters on those scores, and the conversion dropped them — so every YOLO detection reported **1.0**, including one that had just cleared the threshold. Both backends now pass their own scores (`boxes.conf`; `CombineDetections.filtered_confidences`, post-NMS), and a length mismatch **raises** rather than being `zip`ped away, because a score attached to the wrong box reads as a measurement of that box. **`confidence` is now `float \| None`, defaulting to `None`:** `1.0` was a substitute value, the fifth this milestone has deleted, and it made the LoG detector claim certainty it never computed — its blob response is not a probability, and normalising one would be a claim only **M3-T15** could license. Delta: **29 keys added, 0 values changed**. The finding is that `contracts.default_detection_confidence` is `ADDED`: the harness recorded the defaults of D-16's field and not of D-09's, one line below, **so the golden could never have caught this**. 7 tests; restoring the drop turns 6 red. mypy **14 → 12** — the second array would have added a third `_last_result` error, and annotating that field removed all three |
| M3-T06 | Otsu sizing: raise on empty-after-filter; report post-filter `n_objects` | D-05, D-06 | high | [x] Done 2026-08-05, **ADR-0017**. The empty-after-filter case raised nothing and returned `nan`; it now raises and names the parameter, its value and the largest object measured. `n_objects` counts survivors. Delta: **8 golden differences** — `n_objects_reported` **1023 -> 75** on `afm_sparse_low_snr` (13.6x over-count), the `extreme_aspect` degenerate input now fails as `estimate_radius_otsu` instead of as `cannot convert float NaN to integer` in `build_substrate_map`, and 5 keys added for D-05's own reproduction. 4 tests; restoring the old behaviour turns 3 red. **It also broke an M3-T01 test that had been passing on the `nan`** |
| M3-T07 | Guard LoG normalisation against a zero maximum | D-11 | medium | [x] Done 2026-08-05, **ADR-0018**. Both `z_above / z_above.max()` sites now stop on a non-positive or `nan` maximum: the threshold estimator returns the named `DEFAULT_THRESHOLD = 0.05`, `detect_particles` returns an empty `(0, 4)` — zero particles is an answer, not an error (the opposite call from ADR-0017, and the ADR says why). Delta: **65 golden keys added, 0 changed** — the working path is byte-identical, and the number that was wrong (an adaptive threshold of **2.4997** on a negative map, outside the `[0, 1]` it is compared against) had **never been recorded**. Two harness fixes in the same commit made it visible: a `negative_with_structure` degenerate input, and scalars recorded instead of the string `"non-array"`. 11 tests; restoring the raw division turns 3 red |
| M3-T08 | `flatten_lines` must promote dtype like `flatten_plane` does | D-13 | medium | [ ] |
| M3-T09 | Define and apply the opening-radius rounding rule (half-integer radii break `disk()` centring) | D-10 | medium | [x] Done 2026-08-05, **ADR-0020**, decision **B4 → round up**. Every radius reaching `disk()` is `ceil`-ed, in `get_substrate_map` — the funnel all three sites pass through — and `build_substrate_map` reports the integer it used, keeping ADR-0014's principle intact. Up, not down: a radius smaller than a particle recovers a "substrate" containing the particle. Delta: **696 golden values move, 0 keys added** — opening radius +1 or +2 on all five AFM phantoms, **no particle count changes**, largest height move **0.049 nm (0.37 %)** on `afm_dense_overlapping`. mypy **18 → 15**, and the three errors that went were this defect's static shadow. 11 tests; restoring the floor turns 4 red |
| M3-T10 | Detection polarity: TEM currently returns 0 of 22 particles | D-12 | high | [x] Done 2026-08-05, **ADR-0023**, decision **B3 → configured, per-modality default**. `Polarity` (written in M2-T02, adopted by nothing until now) is a `PipelineConfig` field whose `None` resolves to the modality's convention; both detectors take one. One inversion at the entrance (`max - z`, its own inverse and positive-maximum-safe per ADR-0018) and, on the YOLO side, `_prepare_image` inverts only a bright-on-dark image. Delta: **`tem_dark_particles` 0 → 22 of 22 blobs**, prepared YOLO input mean grey 43.3 → 211.7, `config_fields` 12 → 13; **SEM and all five AFM phantoms byte-identical**. Not claimed: better YOLO detections — inference is outside the gate. 14 tests |
| M3-T11 | Handle unknown pixel scale (`None`) without crashing both detectors | D-07 | high | [x] Done 2026-08-05, **ADR-0019**. `None` now propagates and the nanometre value becomes absent: `Detection.radius_nm` is `float \| None`, the LoG blob array's nm column is NaN (an ndarray cannot hold `None`) and `_blobs_to_detections` maps it to `None` at the entity boundary. Pixel-space output is bit-identical with and without a scale. Delta: **168 golden keys added, 0 changed** — every phantom has a scale, so nothing recorded moves; the new keys record a path that used to raise `TypeError`. **mypy 19 → 18**: the error it removes, `pipeline.py:62`, *was* D-07 reported at the assignment instead of at the crash, and had sat in the baseline since M1-T04. 8 tests; substituting the tempting wrong fix `pixel_size_nm or 1.0` turns 4 red |
| M3-T12 | Empty measurements must return a schema-stable DataFrame | D-08 | high | [x] Done 2026-08-06, **ADR-0027**. `pd.DataFrame([])` has **zero columns**, so a scan with nothing measurable answered every read by name with `KeyError`. The baseline schema is now declared — twelve columns, each with a dtype — and returned whether or not a row survived; a test proves the declaration still describes what the **populated** path emits, which is the only thing keeping a declared schema honest. Delta: **78 golden differences, all columns appearing where there were none, 0 values moved**. The sixth block is the finding: `afm_sparse_low_snr` detects **0 blobs on its ordinary path**, so D-08 was live on a real phantom's normal run and the golden had been recording `columns: []` for it since the baseline. 7 tests; restoring `pd.DataFrame(results)` turns 3 red. **Out of scope, and why:** the two SAM2 producers vary their columns *per row* (D-16/D-17 → M3-T14), and `run_pipeline`'s detect-mode empty frame needs the modality-dependent schema T14 decides. **Found while testing:** a NaN height passes `<= 0` and reaches the table — filed as **B-059**, not fixed here (ADR-0010) |
| M3-T13 | Typed error taxonomy + input validation at every numerical entry point | D-15 | medium | [ ] |
| M3-T14 | One measurement schema across all four producers; fix the `bbox` contract | D-16, D-17 | medium | [ ] |
| M3-T15 | Evaluation harness: precision / recall / localisation vs phantom ground truth | — | — | [ ] |
| M3-T16 | Characterize `_read_nanoscope_z` against multiple Nanoscope versions | gap | — | [ ] |
| M3-T17 | `_read_nanoscope_z` divides `None` by `samps` when the header has no `Scan Size` — the fallback branch crashes on the next line (`afm_io.py:95-98`) | **new**, found by mypy in M1-T04 | high | [x] Done 2026-08-06, **ADR-0026**. **D-07 is now closed on all three faces** — detectors (M3-T11), the npy loader (M3-T20), the SPM header (this). A header with no `Scan Size` returns `(None, None, z)`: the height map decodes as always and only the metadata is absent. The same expression's other failure mode went with it — `Samps/line: 0` was a `ZeroDivisionError` naming nothing — and a *stated* non-positive `Scan Size` is now rejected too, which is ADR-0025's absent-versus-wrong distinction applied to the second loader. The annotation stops lying (`-> tuple[float \| None, float \| None, np.ndarray]`): **mypy 15 → 14, and the error that went was this defect's**. Delta: **0 golden differences, and none was possible** — `afm_io` has no phantom, so its 28 unit tests are the whole safety net. 3 tests; restoring the division turns 3 red |
| M3-T18 | `YoloDetector._last_result` is initialised to `None`, so its type is `None`; `.filtered_boxes` is accessed unguarded (`yolo_detector.py:50,87,99`) | **new**, mypy | medium | [x] Done 2026-08-06 **as a side effect of M3-T05**, not as its own commit. Threading the confidences through `_detect_tiled` would have added a *third* `"None" has no attribute ...` error on this field, so it was annotated `Any` — which its own comment already described — and all three went. mypy **14 → 12**. **No runtime guard was added, and none is needed:** every access sits two lines below the assignment inside the same method. The public `last_result` property can still return `None` before the first `detect()`, which is the documented meaning of the field, not a defect. `Any` rather than a real type because both possible classes live in optional heavy dependencies that must not be imported at module level |
| M3-T19 | `estimate_log_threshold_adaptive` rebinds `responses` from `list[float]` to ndarray before calling `.min()`/`.max()` (`log_detector.py:111,116`) | **new**, mypy | low | [ ] |
| M3-T20 | `load_afm(fmt="npy")` fabricates a physical scale: `pixel_size_nm or 1.0` and `scan_size_nm or float(z.shape[0])` (`afm_io.py:132-133`). Unknown scale must be `None` — the invariant D-07 states. Two consequences: every downstream `_nm` becomes a pixel count wearing nanometre units, and `or` also swallows an explicit `0.0`. Row count is used as a length in nm, which is not even dimensionally a size | **new**, found by the M1-T06 tests | high | [x] Done 2026-08-06, **ADR-0025**. **D-07 is now closed on both sides.** The loader passes through what it was given; `AFMRawData` and `PreprocessingResult` carry `float \| None`; a scale that *is* given must be positive, so `0.0`, `-1` and `nan` raise instead of being swallowed by `or`. `build_substrate_map` accepts `None`: the `_nm` outputs are `None` and the `min_size_nm` filter cannot be applied, which is **warned**, never silent (silent is D-04). Delta: **5 golden keys added, 0 values changed** — every phantom has a scale. What the new keys record is the finding: an unscaled run equals a scaled one with `min_size_nm=0`, so on the four clean phantoms the substrate is **bit-identical**, and on `afm_sparse_low_snr` it is **not** — 17 objects become **3351**, the typical radius falls 2.99 px → 0.80 and the opening radius 8 → 5. Losing the scale means losing the filter, and where the filter was load-bearing that reaches the substrate. 10 tests; restoring `pixel_size_nm or 1.0` turns 6 red |
| M3-T21 | `use_tiling=True` — the default backend — produces **exactly one crop**. `_prepare_image` hands `MakeCropsDetectThem` a 640x640 image and the crop shape is also 640x640, so `get_crops_xy` computes `int((640-640) / (640*0.75)) + 1 = 1` step on each axis. The sliding window covers the whole image in a single tile: the tiled backend does the same work as the direct one, more slowly, and small particles are never seen at native resolution — which is the only reason tiling exists. Real tiling needs an input of at least `shape * (2 - overlap/100)` = 1120 px, and a 512x512 scan cannot reach it. Fix is a decision between a smaller crop shape and feeding the scan at native resolution (which then needs D-21's letterbox to survive the library's own resize) | **new**, found while reading for M3-T04 | high | [x] Done 2026-08-05, **ADR-0021**, decision **B7 → keep the backend, stop defaulting to it**. `use_tiling` now defaults to `False` in both `YoloDetector` and `PipelineConfig`; the degenerate case logs what it will do; `_crop_steps` turns the arithmetic into a tested method. Delta: **zero golden difference**, because inference is outside the gate — but the backends are *not* bit-identical even at one crop (`CombineDetections` runs a second NMS), so real detections may move and **no claim is made that either is better**; M3-T15 owns that. Not deleted: the fix is an input-size trade-off nothing can measure until M3-T15 exists. 9 tests |

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
