# Progress

Append-only session log. Newest entry on top. Every session adds one entry, finished or not.

**Entry format:** date · milestone · task IDs · what changed · what was learned · what is next.
A session that changes scientific output states the numerical delta explicitly.

---

## 2026-08-04 — M1 · `M1-T04` Mypy configuration

**Task:** M1-T04 (complete)
**Branch:** `chore/mypy-config`
**Scientific impact:** none — configuration only, no source file edited

### The 30 errors, classified before writing any configuration

The rule for this task was that configuration must not silence a real bug. So all 30
default-run errors were read against the source first.

| Class | Count | Disposition |
|---|---:|---|
| Missing third-party stubs | 8 | silenced per module — pandas, scipy, patched_yolo_infer, ultralytics |
| **Static confirmation of known audit defects** | **13** | kept visible; already have M3 tasks |
| **Real typing defects not previously filed** | **5** | kept visible; **filed as M3-T17…T19** |
| Stub strictness / known stub | 4 | kept visible, harmless |

**The 13 that confirm the audit.** mypy independently reproduces, statically, defects the
Phase 0 audit found by execution:

| Error | Defect |
|---|---|
| `preprocess.py:202  Cannot determine type of "opening_radius"` | **D-01** — the critical `UnboundLocalError`. The manual-radius branch never assigns it; mypy sees the same missing assignment the runtime does. |
| `preprocess.py:149,158  return float, expected int` + `:184 arg-type float→int` | **D-10** — `estimate_rough_radius` is annotated `-> int` and returns a float, which reaches `disk()` and produces even-sized structuring elements. The whole chain is visible in three errors. |
| `types.py:63  tuple[Never, ...] vs tuple[int, int, int, int]` | **D-16** — `bbox` defaults to `()` against a four-element annotation. |
| `preprocess.py:164`, `log_detector.py:125,257`, `pipeline.py:52` | **D-07** — implicit `Optional` and the unknown-pixel-scale contract. |
| `pipeline.py:94 ×2, :110  ndarray \| None where ndarray expected` | the SEM/TEM path, where `z_flat` is `None` by construction. |
| `afm_io.py:100  returns tuple, annotated ndarray` | **D-02** — the return-convention change that silently broke `preprocess_batch.py`. |

A configured type checker would have caught the project's single critical defect before
it was ever committed.

**The 5 that are new.** Filed as tasks, not suppressed:

- **`afm_io.py:98` — new defect, not in the audit.** When the header carries no
  `Scan Size:` field the code sets `scan_size_nm = None` and then immediately evaluates
  `pixel_size_nm = scan_size_nm / samps` → `TypeError`. The `else` branch exists
  specifically to handle that header, and it crashes on the next line. → **M3-T17**
- `yolo_detector.py:50,87,99` — `_last_result` is initialised to `None`, so its inferred
  type is `None`; `.filtered_boxes` is then accessed unguarded. → **M3-T18**
- `log_detector.py:111,116` — `responses` is annotated `list[float]` and then rebound to
  an ndarray before `.min()`/`.max()`. Works at runtime, wrong as a contract. → **M3-T19**

### Configuration

- `[tool.mypy]`: `python_version = "3.12"`, `files = ["src"]`, `warn_unused_configs`,
  `warn_redundant_casts`, `warn_unused_ignores`
- **`nanoscope.*` is strict from its first line** — `disallow_untyped_defs`,
  `disallow_incomplete_defs`, `disallow_untyped_calls`, `disallow_any_generics`,
  `check_untyped_defs`, `no_implicit_optional`, `warn_return_any`, `strict_equality`.
  Retrofitting strictness later is far more expensive than starting with it.
- **`src/` posture: checked, not silenced.** No `ignore_errors`. It carries 22 errors,
  13 of which are the most valuable output this tool has produced; hiding them to make a
  number green would be the opposite of the point. The package is deleted in M2-T15, so
  the errors are a documented baseline, and CI reports them without blocking (M1-T08).
- `ignore_missing_imports` scoped **per module**, never globally — a blanket setting would
  also hide a typo in a first-party import.

`mypy` now runs with no command-line flags: **22 errors in 7 files** (30 minus the 8
stub gaps). It emits one note — `unused section(s): module = ['nanoscope.*']` — which is
deliberate: it is a visible reminder that M2-T01 has not happened yet, and it disappears
the moment the package exists. Verified non-fatal: mypy exits 0 on a clean file with that
note present.

### Considered and rejected

Installing `pandas-stubs` and `scipy-stubs` instead of silencing those imports. It would
give real coverage, but pandas-stubs against pandas 2.x typically produces a fresh wave of
errors in code that is scheduled for deletion in M2. Revisit when `nanoscope` exists —
backlog **B-057**.

### Next

`M1-T05` — wire the characterization golden into `pytest`. It is the only check that
passes today and the only protection M2 has, and it currently runs only when someone
remembers to type the command.

---

## 2026-08-03 — M1 · `M1-T03` Ruff configuration repair

**Task:** M1-T03 (complete)
**Branch:** `chore/ruff-config`
**Scientific impact:** none — `capture.py` reports `characterization baseline stable (9 groups)`

### Done

- **Removed `fix = true` / `show-fixes`.** `ruff check .` was rewriting source files as a
  side effect of being asked a question — 66 automatic edits to the scientific core, from
  a command the documentation told people to run. Fixing is now explicit: `ruff check --fix`.
- Moved `select` / `ignore` under `[tool.ruff.lint]`; the deprecation warning is gone
  (stderr is now empty).
- `target-version` `py311` → `py312`; the project requires `>=3.12`.
- `known-first-party` `["your_package_name"]` → `["src"]` (unedited template value; it
  becomes the real package in M2-T01).
- `classmethod-decorators`: dropped `pydantic.validator` — pydantic is not a dependency.
- `per-file-ignores`: dropped `S101`; the `S` (bandit) family is not selected, so the
  entry was dead configuration. Backlog **B-056**.
- Excluded `*.ipynb` from lint. Notebooks are experiments, not interfaces
  (PROJECT_RULES §7); their 68 findings are import order and prints in exploratory cells.
  Notebook hygiene is M1-T09.

### Verification

| Check | Result |
|---|---|
| Config deprecation warning | gone — stderr empty |
| `ruff check .` modifies the tree | **no** — `git diff --exit-code` clean after every run |
| `src/` findings before vs after | **identical** — `--statistics` diff is empty |
| Total findings | 196 → **128** (the 68 excluded are all notebooks) |
| `ruff format --check .` | runs; 18 files would be reformatted (not fixed here — M2) |
| Characterization | zero drift |

### Correction to the M1-T02 baseline

The `src/` figure recorded in M1-T02 was **109, not 108**. I produced the 108 by summing
a `--statistics` listing that I had truncated with `head -20`, dropping the last row
(`W291 trailing-whitespace`, 1). The commit message of `13857e5`, and `STATE.md` /
`TASKS.md` before this entry, carry the wrong number.

The distribution is otherwise unchanged, and the invariant this task was checked against
still holds: the configuration repair changed **nothing** about what is reported — the
before/after statistics diff is empty. Corrected everywhere in the living documents;
the commit message of `13857e5` is history and stays as written.

Burn-down target for M2 is therefore **109 findings in `src/`**, of which 44 are the
ambiguous-unicode signature of the Russian text (D-22) and 13 are `print` (D-23).

### Next

`M1-T04` — mypy configuration. 30 errors today with default settings; the task is to
choose strictness for new code and a baseline exclusion for `src/` until M2 lands, not to
fix the errors.

---

## 2026-08-03 — M1 · `M1-T02` Dev dependencies and quality baseline

**Task:** M1-T02 (complete)
**Branch:** `chore/dev-dependencies`
**Scientific impact:** none — `capture.py` reports `characterization baseline stable (9 groups)`
after the environment change

### Done

- `[dependency-groups] dev` added to `pyproject.toml`: pytest, pytest-cov, ruff, mypy
- `uv sync`; `uv.lock` gained **only** tooling packages — no runtime version moved.
  Verified against the golden `_meta` pins: torch 2.7.1+cu118, NumPy 2.4.4, SciPy 1.17.1,
  scikit-image 0.26.0 — all unchanged, which is why the golden is still stable

| Tool | Version |
|---|---|
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |
| ruff | 0.16.1 |
| mypy | 2.3.0 |

### Baseline — the M2 burn-down target

Measured, nothing fixed. These are the numbers M2 has to drive to zero.

**ruff — 196 findings total, 109 in `src/`** (66 auto-fixable)
*(corrected in the M1-T03 entry above; this session recorded 108 from a truncated listing)*

| Rule | src/ | What it is |
|---|---:|---|
| RUF002/003/001 | **44** | ambiguous unicode in docstrings, comments, strings — this is the Russian text of **D-22**, found mechanically |
| T201 | **13** | `print` in library code — **exactly the 13 the audit counted by hand** |
| F401 | 11 | unused imports |
| I001 | 11 | unsorted imports |
| W293/W292/W291 | 10 | whitespace |
| RET504/505 | 7 | unnecessary assign / superfluous else |
| RUF046 | 2 | unnecessary `int()` cast — **adjacent to D-10**, the opening-radius rounding defect |
| RUF013 | 2 | implicit `Optional` — the unknown-scale contract (**D-07**) |
| A005 | 1 | `src/types.py` shadows the stdlib `types` module — a real M2-T02 constraint |
| others | 8 | B007, C408, N806, PIE790, RUF022, SIM108, UP037 ×2 |

The remaining 88 findings are in notebooks, `preprocess_batch.py` and
`tests/characterization/capture.py`.

**mypy — 30 errors in 9 files** (default settings, no configuration yet)

| Code | Count |
|---|---:|
| import-untyped | 8 |
| assignment | 7 |
| arg-type | 6 |
| return-value | 3 |
| attr-defined | 3 |
| has-type, empty-body, call-overload | 3 |

The most interesting one is static confirmation of a known runtime defect:

```
src/pipeline.py:110: error: Argument 4 to "run_sam2_from_blobs" has incompatible type
"ndarray[...] | None"; expected "ndarray[...]"
```

That is the SEM/TEM path, where `z_flat` is `None` by construction (`pipeline.py:53`).
A type checker would have caught it before it ever ran.

**pytest — 1 test, 1 failed**

```
FAILED tests/test_io.py::test_load_spm - FileNotFoundError: 'data/5.011'
```

Correction to the prediction in `CURRENT_TASK.md`: the test does not pass vacuously, it
**fails**. It catches `ImportError` while `load_afm` raises `FileNotFoundError` (audit
D-20), and the path is absent from a clean checkout. The suite has been red the whole
time; nobody could see it because pytest was never installed. M1-T06 replaces it.

### Side effect worth knowing about

`uv sync` uninstalled three packages that were in the environment but not in `uv.lock`:
`clip` (from the ultralytics CLIP repo), `ftfy`, `regex`. They were installed outside uv,
so `uv sync` removed them to match the lock — expected behaviour, but not something I
intended.

Nothing under `src/` imports them; they are needed only for **YOLO-World** models, and
`checkpoints/yolov8s-world.pt` is such a model (it is not the configured default —
`PipelineConfig.yolo_model_path` points at `best12x.pt`). If YOLO-World is wanted, the
fix is to declare it as a real dependency rather than let it be installed ad hoc:

```bash
uv add "clip @ git+https://github.com/ultralytics/CLIP.git"
```

Recorded as backlog **B-055**.

### Not done, deliberately

No finding was fixed. Repairing the ruff configuration is M1-T03, the mypy configuration
is M1-T04, and the 109 `src/` findings are M2 work — under the protection of the golden
file, not before it.

### Next

`M1-T03` — repair the ruff configuration. It currently emits a deprecation warning for
top-level `select`/`ignore`, targets `py311` on a 3.12 project, still carries
`known-first-party = ["your_package_name"]`, and — most importantly — sets `fix = true`,
so `ruff check .` rewrites source files as a side effect of being asked a question.

---

## 2026-08-03 — M1 · `M1-T01` Repository hygiene

**Task:** M1-T01 (complete), M1-T11 (complete — absorbed)
**Branch:** `chore/repo-hygiene`
**Scientific impact:** none — `capture.py` reports `characterization baseline stable (9 groups)`

### Result

| Metric | Before | After |
|---|---:|---:|
| Tracked files | 2 877 | **77** |
| Tracked under `frontend/node_modules` | 2 800 | **0** |
| Tracked model weights | 1 (`yolov8s-world.pt`, 26 MB, staged) | **0** |
| Largest tracked non-notebook file | 6.5 MB | 3.2 MB (README figure) |

### Done

- `git rm -r --cached frontend/node_modules` — 2 800 files untracked, all still on disk
- `git rm --cached yolov8s-world.pt` — the 26 MB blob (`2fa1b38`) was staged for addition
  and would have entered history on the next `git commit` without a pathspec. Removed
  from the index before that happened; the file is not on disk and the four real
  checkpoints under `checkpoints/` are untouched
- `.gitignore` rewritten: added `node_modules/`, `output/`, `*.pt`, `*.pth`, `*.onnx`,
  `*.safetensors`, `*.zip`, `*.tar.gz`, `build/`, `dist/`, `*.egg-info/`, `.mypy_cache/`,
  `.coverage`, `htmlcov/`; grouped and commented by rationale
- `.claude/settings.json` now **tracked** — agent configuration is shared (PROJECT_RULES §7).
  `.claude/settings.local.json` (per-machine permissions) stays ignored. This completes
  the `.gitignore` edit that was already sitting uncommitted in the working tree
- **Junk removed from disk:**
  - `.zip` — a 22-byte *empty* zip archive, tracked since February
  - `__pycache__/` × 4, including `.pyc` files for modules that no longer exist
    (`sam2_pipeline`, `config`, `detection`) and bytecode from CPython 3.14 while the
    venv is 3.12 — stale in two independent ways
  - `.pytest_cache/`, `.ruff_cache/`
  - `output/`, `notebooks/` — both empty directories
  - root `package-lock.json` — an empty stray from an accidental `npm install` at the
    repository root (the real one is `frontend/package-lock.json`)
- `plan.md` → `docs/archive/plan-frontend-react-client.md`, un-ignored and now tracked,
  with an ARCHIVED header pointing at ADR-0007. It was gitignored, so the only record of
  the intended HTTP contract was unshareable. Path references in ADR-0007 updated
  (editorial only — the decision is unchanged)

### Deviation from the stated Definition of Done

"Largest tracked file, excluding the pre-existing notebooks, < 1 MB" is **not met**:
`images/yolo_sam2_comparison.png` (3.2 MB) and `images/log.png` (3.0 MB) are README
figures. They are legitimate content, not junk, and untracking them would break the
README. Recorded as backlog **B-054** (optimise figures) rather than silently ignored.

The notebooks (6.5 MB + 2.2 MB, committed with outputs) are untouched — that is M1-T09.

### Not done, deliberately

History still carries the 78 MB: `git rm --cached` stops the growth, it does not shrink
`.git` (still 81 MB). Rewriting history invalidates every clone and needs the operator's
approval — backlog **B-040**.

### Next

`M1-T02` — declare `pytest`, `pytest-cov`, `ruff`, `mypy` as dev dependencies. None of
them is installed today, so the quality gate does not exist yet; the characterization
runner is currently the only working check.

---

## 2026-08-03 — M0 · Engineering foundation

**Tasks:** M0-T01 … M0-T08 (all complete)
**Branch:** `docs/engineering-infrastructure`
**Base:** `11e0ecc` (frontend init)
**Code changed:** none — documentation only, by design

### Done

- Read `systempromt.md`, `PROJECT_CONTEXT.md`, `README.md`, plus `project.md`, `plan.md`,
  the Phase 0 audit and the characterization baseline
- Analysed the repository directly: 12 modules / 2 021 LOC under `src/`, 13 frontend
  source files, 8 characterization phantoms, 2 854 tracked files
- Recorded 7 strengths and 16 weaknesses with measured evidence → `Architecture.md` §2
- Defined the target Clean Architecture: `app` / `core` / `application` / `infrastructure` /
  `gui` / `resources`, with an enforced dependency rule and a layer-contract table
- Wrote the constitution → `PROJECT_RULES.md`
- Broke the project into 10 milestones (M0–M9) with exit criteria → `Roadmap.md`
- Broke the milestones into 110 tasks → `TASKS.md`
- Wrote 11 ADRs (0001–0011) → `docs/ADR/`
- Established the state protocol → `STATE.md`, `CURRENT_TASK.md`, this file
- Selected `M1-T01` as the first task

### Learned

- **The starting position is better than a greenfield.** A completed, *reproduced* Phase 0
  audit and a committed golden baseline over 8 seeded phantoms already exist. That changes
  the strategy: the domain can be moved aggressively, because drift is detectable to
  `rtol=1e-6`.
- **The stack pivoted.** The previous direction was React + a FastAPI backend that was
  never written; the target is a Qt6 desktop application. The React client is the only
  work made obsolete by the pivot, and it is parked rather than deleted (ADR-0007).
- **The domain layer is genuinely worth preserving.** A modality-neutral `Detection`, a
  `BaseDetector` ABC, lazily imported SAM2 and a deliberate dependency root mean the
  Clean Architecture target is an extraction, not a rewrite.
- **Two problems are urgent for different reasons.** `node_modules` (98% of tracked files)
  makes review impossible; the staged `yolov8s-world.pt` has a closing window before it
  is permanently in history. Both are M1-T01.
- **Structure must precede correctness.** Fixing D-03 or D-04 today would change numbers
  inside a codebase with 5 import cycles and no test gate. M2 before M3 is not
  bureaucracy — it is the only way the deltas stay attributable.

### Open questions raised

B1 package name · B2 `min_size_nm` semantics · B3 detection polarity · B4 opening-radius
rounding · B5 fate of `frontend/` and the notebooks · B6 real sample data in git.
Full text in `STATE.md`. None blocks M1.

### Next

Execute `M1-T01` on branch `chore/repo-hygiene`: untrack `frontend/node_modules` and
`yolov8s-world.pt`, rewrite `.gitignore`. See `CURRENT_TASK.md`.

---

## Before 2026-08-03 — inherited context

Not a session log; recorded so the history is not lost.

| When | What |
|---|---|
| 2026-07-28 | Phase 0 audit: 24 defects reproduced by execution, 5 import cycles, 10 dead functions → `docs/audit/2026-07-28-baseline-audit.md` |
| 2026-07-28 | Characterization baseline: 8 seeded phantoms, golden file at `rtol=1e-6` → `tests/characterization/`, `docs/audit/characterization-baseline.md` |
| `11e0ecc` | React + Vite frontend scaffolded against an unimplemented `/analyze` backend |
| `e8caf25` | `afm_io` reworked to return `AFMRawData` (silently broke `preprocess_batch.py` — D-02) |
| `cd360aa` | Generalisation to SEM/TEM: `MicroscopyData`, `load_microscopy_image` |
| `f1cf175` | `types.py` and `pipeline.py` introduced — the first deliberate layering |
| `0ef8c50` | Detection refactored to `BaseDetector` + `LogDetector` / `YoloDetector` |
| earlier | SAM2 integration, tiled YOLO, LoG baseline, morphological substrate estimation |
