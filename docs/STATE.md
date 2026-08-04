# STATE

**Last updated:** 2026-08-04 · **Branch:** `feat/delete-src-shims` · **Base commit:** `8229f06`

> This file is mandatory and must be updated at the end of **every** development session.
> Read it first when a session starts.

---

## Current milestone

**M3 — Numerical correctness**

Fix the defects the audit reproduced. **The rules change here:** every task gets its own
commit, its own ADR, its own golden update, and a quantified before/after delta in
`docs/Progress.md`. Never bundled (ADR-0010).

**M2 closed 2026-08-04** — sixteen tasks, 2 021 lines of science moved into four named
layers, and **not one number changed**. Five of six exit criteria met in full; the sixth
(ports) is partly met on purpose and M4 owns the rest. Milestone summary in
`docs/Progress.md`.

**M1 closed 2026-08-04** — all eleven tasks done, four of five exit criteria met. The
fifth (no tracked file over 1 MB) has two known exceptions, the README figures, filed as
**B-054** and deferred to M9-T01. Milestone summary in `docs/Progress.md`.

## Current task

**`M3-T01` — fix `build_substrate_map(manual_radius_px=...)`.** Status: **in progress**.
`opening_radius` is never assigned on the manual branch, so the call raises
`UnboundLocalError` **100% of the time** (D-01, critical). The golden already records that
exception, so the fix appears as a declared change rather than a surprise. First numerical
change in the project.

---

## Completed

### M2 — Domain extraction ✅ (closed 2026-08-04)

- **M2-T01** ✅ (2026-08-04) — `nanoscope/` exists: the six layers from ADR-0011
  (`app` `core` `application` `infrastructure` `gui` `resources`) plus `py.typed`, each
  `__init__` stating that layer's half of the dependency rule. Distribution renamed
  `afm-analysis` → `nanoscope`. The regenerated `uv.lock` was **diffed package by package
  before committing** — 119 shared packages, **0 version changes** — because CI runs
  `uv sync --locked` and a quiet re-resolution of numpy or scipy would move the golden for
  a reason unrelated to the task. mypy now checks 20 files instead of 13, and the strict
  `nanoscope.*` override that M1-T04 wrote before the package existed **binds for the
  first time**: 0 errors, strict from line one. **Zero code moved**; no sub-package below
  the layer level, since each arrives with its content in M2-T02…T08.

- **M2-T02** ✅ (2026-08-04) — the first scientific code to move, in **three commits** so
  that drift would be attributable without bisecting. The six dataclasses left
  `src/types.py` for `nanoscope/core/entities/`; `src/types.py` is now a shim that defines
  nothing, verified by loading the pre-move module beside the new one — identical fields,
  order, defaults and factories, and `src.types.X is nanoscope.core.entities.X` for all
  six. **One `Detection` class in the process, not two.** The strict `nanoscope.*` override
  then caught three things verbatim legacy code does not satisfy: two bare generics
  tightened to `dict[str, Any]`, and `Detection.bbox` given a scoped `type: ignore` —
  mypy complaining there *is* **D-16**, and fixing it moves a number the golden records, so
  M3 owns it; `warn_unused_ignores` makes the ignore expire itself. **nanoscope: 0 mypy
  errors.** Finally `Modality`, `Polarity`, `PixelScale`, `DeviceKind` with 8
  mutation-validated tests — **defined, adopted by nothing**, because adoption changes what
  `asdict` produces. Golden: **zero drift**.

- **M2-T03** ✅ (2026-08-04) — preprocessing moved to
  `nanoscope/core/science/preprocessing/` (`flatten.py` + `substrate.py`); `preprocess.py`
  is a shim. The first move of real behaviour — plane fitting, line detrending,
  morphological opening, Otsu. **Proved before the gate ran:** all six functions
  AST-identical, docstrings differing only in trailing whitespace, and the 5 mypy errors
  travelled with the code (21 before, 21 after). Golden: **zero drift**. What the task
  actually settled is how legacy enters a strict package: **declared once in configuration**
  — mypy at default strictness for `nanoscope.core.science.*`, ruff still blocking there
  but ignoring six named rules — instead of a `type: ignore` on every audited defect across
  fifteen more moves. Every entry names the task that deletes it (M2-T11, M2-T12, M3).

- **M2-T04…T06** ✅ (2026-08-04) — three tasks on one branch (they share shims), **16
  definitions moved, golden zero drift**. I/O split along parsing-versus-the-world:
  SPM decoding to `core/science/io/`, the path-opening functions to
  `infrastructure/storage/`. The LoG detector and its ABC to `core/science/detection/` —
  all 7 definitions AST-identical, and `detect_particles` is recorded for all 8 phantoms.
  Measurement split AFM height from mask geometry, which is the point of M2-T06: the
  modality-neutral code was trapped in an AFM module, so the SEM/TEM path depended on
  `src.measure` by accident. Four more `src/` modules are shims. **The `ImageLoader` port
  was deliberately not written** — M2-T08 defines the ports wholesale. mypy 21 → 21.
  Three of ruff's safe fixes landed in `loaders.py` and are named, not rounded up to
  "verbatim". **`RUF046` was wrong about the science**: `round(np.float64)` is not an int,
  so obeying it would have changed the dtype of every measurement DataFrame's `x_px`
  column — it is now ignored with that reason attached.

- **M2-T07 / M2-T08** ✅ (2026-08-04) — the model-backed code left the domain:
  `YoloDetector` → `infrastructure/models/`, the SAM2 runners beside it, and
  `afm_to_rgb`/`overlay_masks` → `infrastructure/imaging/` (neither ever belonged to
  SAM2). **Nothing under `core` imports torch, ultralytics, sam2 or patched_yolo_infer any
  more** — the dependency rule is now a fact, and a test asserts it against `sys.modules`.
  `F821` caught two dangling references the split created, before any test ran. mypy 21 →
  21, after both moved modules joined `core.science` at default strictness. **M2-T08 was
  narrowed on purpose: one port, not seven.** `Detector` is satisfied today by
  `LogDetector` and `YoloDetector` from opposite layers; the other six have no
  implementation and no caller, so they ship with their first adapter, and
  `core/ports/__init__.py` carries the table naming the task for each.

- **M2-T09 / M2-T10** ✅ (2026-08-04) — the layout became enforceable and the rules became
  executable. **All five import cycles (D-18) had one cause**: `src/__init__.py` re-exported
  the pipeline, and Python runs a package `__init__` first, so importing the "dependency
  root" loaded SAM2 and matplotlib. Nothing ever used `from src import X` — emptying one
  file fixed all five. `import src.types` **1198 → 187 modules, 0.77 s → 0.07 s**;
  `nanoscope.core.entities` **626 → 185**, pandas moved behind `TYPE_CHECKING`.
  `test_import_graph.py` checks direction statically over the AST and weight dynamically in
  a subprocess; both proven to fail. **The M2 exit criterion "< 100 modules" was
  unachievable** — numpy alone is 141 — and is corrected in `Roadmap.md` to a named
  heavy-import assertion plus a 250 bound. M2-T10 put the capability matrix in
  `application/capabilities.py` and **fixed D-14**: validation now runs before any detector
  is constructed, with byte-identical messages. 12 tests carry it, because the golden never
  calls `run_pipeline`.

- **M2-T11…T14** ✅ (2026-08-04) — the library stopped printing, started speaking English,
  shed four dead functions and became installable. **Zero numbers moved**, and for the first
  time that took a *declared* golden re-baseline: 6 changed lines, none of them numeric —
  4 translated exception messages plus `stdout_lines` 8→0 and 4→0, because the golden
  records how much a function prints. **It also caught a bug in M2-T11 before any human
  did** (`"1%%"` is only an escape when `logging` formats, which it does not without args).
  **No `LogSink` port — ADR-0013**: it would only wrap `logging`, whose `Handler` is already
  the extension point. That is the second of seven planned ports to dissolve on contact with
  reality. **M2-T13 deleted 4 of the audit's 10 "unreachable" functions and kept 6** —
  `estimate_log_threshold` is recorded by the golden, `load_microscopy_image` is the only
  SEM/TEM entry point, three are used by the notebooks. `nanoscope` is now a real wheel
  (`py.typed` in, `src/` out) and the `pythonpath` hack is half deleted. Ruff findings inside
  `nanoscope/` with ignores off: **64 → 13**.

- **M2-T15 / M2-T16** ✅ (2026-08-04) — **`src/` deleted entirely**, and the milestone with
  it. The title understated the task: three modules had never had a shim and had to move
  first (`pipeline` and `preprocessing_pipeline` → `application/use_cases/`, `visualization`
  → `infrastructure/imaging/`). `pythonpath` deleted outright; mypy points at one package.
  A test caught a naming trap a review would not have: a module and a function of the same
  name shadow each other through `__init__`. M2-T16 rewrote `PROJECT_CONTEXT.md`, which had
  drifted to describing `src/`, the deleted frontend and a `pytest.ini` removed in M1-T05.

### M1 — Repository hygiene ✅ (closed 2026-08-04)

- **M1-T01** ✅ (2026-08-03) — tracked files 2 877 → **77**; `frontend/node_modules`
  (2 800 files) untracked; `yolov8s-world.pt` (26 MB) removed from the index before it
  entered history; `.gitignore` rewritten; `.claude/settings.json` now shared; junk
  deleted (`.zip`, four `__pycache__/`, tool caches, empty `output/` and `notebooks/`,
  stray root `package-lock.json`); `plan.md` archived to `docs/archive/`.
  Characterization: **zero drift**.
- **M1-T11** ✅ — absorbed into M1-T01.
- **M1-T02** ✅ (2026-08-03) — pytest 9.1.1, pytest-cov 7.1.0, ruff 0.16.1, mypy 2.3.0
  declared and installed; no runtime version moved, golden still stable. Baseline
  measured: **196 ruff findings** (109 in `src/`), **30 mypy errors**, **1 test, failing**.
  Nothing fixed — that is M1-T03/T04 and M2.
- **M1-T03** ✅ (2026-08-03) — ruff configuration repaired: `fix = true` removed (it made
  `ruff check` rewrite sources), `select`/`ignore` moved under `[tool.ruff.lint]`,
  py311 → py312, template `known-first-party` fixed, dead `S101` dropped, notebooks
  excluded from lint. `src/` findings unchanged at 109 — a repair, not a rule change.
  Total 196 → 128. Characterization: **zero drift**.
- **M1-T04** ✅ (2026-08-04) — mypy configured: strict for `nanoscope.*` from its first
  line; `src/` checked but **not** silenced (22 errors after per-module stub handling).
  All 30 default errors classified before writing config: 13 statically confirm audit
  defects **D-01, D-02, D-07, D-10, D-16**, and **3 new defects** were found and filed
  (**M3-T17…T19**), including a crash in the SPM parser's no-`Scan Size` fallback.
- **M1-T05** ✅ (2026-08-04) — the characterization golden now runs under `pytest`, via a
  single new seam in `capture.py` (`diff_against_golden()`); the CLI is unchanged. Marked
  `slow` (**192 s measured**, not the ~100 s the docs claimed); `pytest -m "not slow"`
  skips it in 1.4 s. `pytest.ini` folded into `pyproject.toml` and deleted — while it
  existed, pytest ignored `[tool.pytest.ini_options]` silently. The negative case was
  proven, not assumed: a perturbed golden produced a red run naming the moved quantity.
  **The M2 safety net is now mechanical.**
- **M1-T06** ✅ (2026-08-04) — `tests/test_io.py` (no assertions, wrong exception, absent
  fixture path) deleted; replaced by `tests/unit/test_afm_io.py`: **22 tests** over a
  synthetic Nanoscope byte stream derived from a real local header — round trip,
  calibration, unit conversion, 8 failure modes, npy and SEM/TEM. No binary fixture, no
  `data/`. **`pytest` is green for the first time (23 passed, 200 s).** The suite was
  validated by mutation: 4 edits to the parser, 3 killed immediately, and the 4th exposed a
  test that could not fail — now fixed. One new defect found → **M3-T20**.
- **M1-T07** ✅ (2026-08-04) — pre-commit: **9 hooks, each demonstrated failing** on a
  deliberately bad staged file. ruff runs as a `repo: local` hook on the project's own
  version, so no second version is ever declared. Rewriting hooks (format, whitespace) skip
  `src/` **and `preprocess_batch.py`** — the `--all-files` sweep caught them editing the
  scientific core, which the original `^src/` exclusion missed; refusing hooks apply
  everywhere. pytest and mypy stay off the commit path by design. `src/` files modified:
  **0**; golden: zero drift.
- **M1-T08** ✅ (2026-08-04) — CI written and verified locally: format → lint → tests+golden,
  `src/` reported not blocking. CI installs a `ci` group with **no torch, ultralytics, sam2
  or patched-yolo-infer** — every heavy import in `src/` turned out to be function-local —
  and a step fails the job if one appears. Two traps caught by running it: `uv run` re-syncs
  and would have reinstalled the full runtime (`UV_NO_SYNC` set), and `ruff format` rewrites
  Python inside Markdown docs (`*.md` excluded). The legacy exclusion moved into
  `pyproject.toml`, declared once for hooks and CI. Both rejection cases confirmed red.
  **Then it was pushed, and three runs found what local verification could not:** no
  readable failure reason (job logs need admin → diagnostics added), a non-existent
  `setup-uv@v9` tag (my error; both actions now pinned exactly), and — the real one — a
  single golden difference that was an exception *message*, not a number. CI resolved
  **Python 3.14**, which reworded `too many values to unpack`; 3.12 is now pinned and
  asserted. **Run 4 is green.** The underlying fragility — the golden stores CPython
  exception text — is filed as **B-058** and needs an ADR, not a quiet edit.
- **M1-T09** ✅ (2026-08-04) — notebook outputs stripped with the configured hook:
  **8.3 MB → 32 KB**, every one of the 45 code cells intact, and the outputs remain in git
  history. Both notebooks moved to `notebooks/` with a README stating they are experiments,
  that nothing may import them, and how to recover the outputs. `main.ipynb` — a tracked
  **0-byte file that was not valid JSON** (audit §330) — deleted. Tracked working tree
  17 MB → **7.8 MB**. **`pre-commit run --all-files` is green for the first time**; the
  last red was a missing final newline in one archived document.
- **M1-T10** ✅ (2026-08-04) — one gate, one description: a 53-line `Makefile`
  (`check` `format` `lint` `test` `fast` `golden` `types` `lint-legacy`; bare `make` lists
  them), and **CI rewritten to call the targets** — the workflow no longer holds a copy of
  any command, which was the point. Proven to fail closed: a misformatted file stopped
  `check` at step 1 in **0.04 s**, exit 2, never reaching the 190 s test step; a failing
  test failed its target. `types`/`lint-legacy` stay outside `check` because the legacy
  baseline is non-zero by design — a gate that cannot pass is a gate people skip. Writing
  it exposed that the three existing descriptions had already drifted: `PROJECT_RULES` §6
  listed `mypy nanoscope` (no such package yet) and a golden command M1-T05 had folded
  into `pytest`. **CI run 14 green on the first try, 216 s**, environment assertion intact.
  **M1 closes here.**

### Decisions executed (2026-08-04)

- **B1 → `nanoscope`** — ADR-0011 Accepted. Unblocks every M2 task.
- **B5 → delete** — **ADR-0012** (supersedes ADR-0007): `frontend/` and
  `preprocess_batch.py` removed. Tracked files **78 → 63**, and the blocking lint/format
  carve-out shrank from two paths to one, `src/`, which M2 then dissolves. Ruff findings in
  the legacy core **117 → 109**, all now in `src/`. Both files remain in git history.

### M0 — Engineering foundation (2026-08-03)

- Repository analysed: 12 source modules / 2 021 LOC, plus a React client, notebooks and
  an existing Phase 0 audit
- Strengths and weaknesses recorded with evidence → `docs/Architecture.md` §2
- Target Clean Architecture defined (`core` / `application` / `infrastructure` / `gui` / `app`)
- Project constitution written → `docs/PROJECT_RULES.md`
- 10 milestones, 110 tasks → `docs/Roadmap.md`, `docs/TASKS.md`
- 11 ADRs written → `docs/ADR/`
- Session/state protocol established → this file, `docs/Progress.md`, `docs/CURRENT_TASK.md`
- First task selected → `M1-T01`

### Inherited from earlier work (pre-M0, already in the repository)

- Working scientific pipeline: SPM I/O, flattening, substrate estimation, LoG and YOLO
  detection, SAM2 segmentation, height measurement
- Phase 0 audit with 24 reproduced defects → `docs/audit/2026-07-28-baseline-audit.md`
- Characterization golden baseline with 8 seeded phantoms →
  `docs/audit/characterization-baseline.md`, `tests/characterization/`

---

## In progress

**M2 is closed.** `M3-T01` is in progress on the same branch as M2-T15/T16.

**Repository state:** `main` is at `8229f06` and carries all of M0, M1 and M2-T01…T14.
M2-T15/T16 are on `feat/delete-src-shims`. CI on `main` is green: **216 s**, of which `make test`
is 194 s, and the environment assertion (Python 3.12 + CPU-only) passes, so the green is
green for the right reason.

**There is no `src/`.** One package, `nanoscope`, 41 modules across four layers, installed
rather than path-hacked. That is the shape every remaining
M2 move leaves behind, until M2-T15 deletes them wholesale.

**Legacy in transit is declared, not hidden.** `nanoscope.core.science.*` runs at mypy's
default strictness and carries six named ruff ignores; every entry names the task that
deletes it (M2-T11, M2-T12, M3). The rest of `nanoscope` stays strict and 0.

Locally, `make check` is green end to end: format, lint, then 23 tests including the
golden, exit 0.

---

## Blocked / needs decision

Decisions only the operator can make. Each blocks a specific task.

| # | Question | Blocks | Why it needs the operator |
|---|---|---|---|
| B2 | **`min_size_nm` semantics (D-04).** `int(5 / 9.77) == 0` disables the noise filter on 90% of your scans. What *should* the minimum particle size mean at coarse pixel scales — a floor of 1 px, a rounded value, or an error? | M3-T02 | It defines what counts as a particle; that is physics, not engineering |
| B3 | **Detection polarity (D-12).** TEM particles are dark on bright; the detector keeps the bright side and finds 0 of 22. Explicit configuration per modality, or auto-detection? | M3-T10 | Determines whether TEM support is a setting or a heuristic |
| B4 | **Opening-radius rounding (D-10).** Half-integer radii produce an even-sized structuring element with no centre pixel, shifting `z_result` by half a pixel. Round up, round to nearest odd, or floor? | M3-T09 | Changes substrate estimation on real data |
| B6 | **Real sample data in git.** `data/` holds 628 SPM scans and is ignored. Should one small representative scan be committed as a test fixture? | M3-T16 | Data ownership and repository size |

**Closed 2026-08-04 by the operator:**

- **B1 — package name → `nanoscope`.** ADR-0011 moves from Proposed to **Accepted**. This
  was the last thing blocking **M2**; M2-T01 can start as soon as M1 closes.
- **B5 — fate of the parked work → delete.** `frontend/` (21 tracked files) and
  `preprocess_batch.py` removed under **ADR-0012**, which supersedes ADR-0007. The third
  part of B5, the notebooks, was answered differently in M1-T09: kept, stripped, moved.

None of the remaining questions blocks M1 or M2.

---

## Next

1. **Execute `M2-T02`** — entities and value objects out of `types.py`. Rewrite
   `docs/CURRENT_TASK.md` for it. First moved code, first real use of the golden as a gate
2. `make types` joins `make check` as blocking once enough of `nanoscope/` is real — the
   one deviation recorded against M1's exit criteria. `nanoscope` is at 0 mypy errors and
   strict today, so the only thing keeping `types` out of `check` is `src/`
3. Before any Python upgrade, deal with **B-058** — the golden compares CPython exception
   text, so a new interpreter reads as characterization drift
4. **B-054** (two README figures over 1 MB) is the one M1 exit criterion left open;
   it belongs to the README rewrite in M9-T01, not to M2

---

## Health indicators

| Indicator | Value | Target | Source |
|---|---|---|---|
| Tracked files | **104** (was 2 854) | see note | `git ls-files \| wc -l` |
| Tracked working tree | **7.6 MB** ✅ (was 17 MB) | — | `git ls-files -z \| xargs -0 du -ch` |
| Tracked model weights | **0** ✅ (was 1) | 0 | `git ls-files '*.pt'` |
| `.git` size | 81 MB | — | `du -sh .git` — history unchanged, see B-040 |
| Library LOC | 2 021 | — | `wc -l src/**/*.py` |
| Meaningful tests | **119, all passing** ✅ (was 1, failing) | ≥ 80% of core | `pytest -q` |
| Golden enforced automatically | **yes** ✅ (was: by discipline) | yes | `pytest` |
| `src/` modules moved into `nanoscope/` | **12 of 12** ✅ — `src/` deleted | 12 | `git ls-files` |
| ruff findings, declared-and-owned | **14** in `nanoscope/` (was 109 in `src/`) | 0 | `make lint-legacy` |
| ruff findings, blocking | **0** ✅ | 0 | `make lint` |
| mypy errors | **20**, all inherited with moved code, none silenced; new code strict | 0 | `make types` |
| Characterization phantoms | 8 | 8 | `tests/characterization/` |
| Open defects | 28 (24 audit + 3 mypy + 1 found by the M1-T06 tests) | 0 critical | audit §2, M3-T17…T20 |
| Import cycles | **0** ✅ (was 5), and a test refuses new ones | 0 | `tests/unit/test_import_graph.py` |
| `print` calls in library code | **0** ✅ (was 13), asserted per module | 0 | `tests/unit/test_logging.py` |
| Non-English lines in library code | **0** ✅ (was 197) | 0 | `grep -rn "[а-яА-ЯёЁ]"` |
| Lint/type/test gate | **green end to end** ✅ — hooks on commit, CI on push | stays green | GitHub Actions |
| The gate has one definition | **yes** ✅ — `make check`, and CI calls the same targets | one | `Makefile` |
| Tracked files over 1 MB | **2** ❌ — two README figures, B-054 | 0 | `git ls-files` + `ls -l` |

> **The `< 100` target has done its job and expired — the count passed it at M2-T07.** It was M1's measure of
> *junk* — 2 800 `node_modules` files. M2 adds real source: each move leaves a shim and
> creates two or three modules. Passing 100 means the extraction is working, not that
> hygiene regressed. The
> meaningful successor is the row above it: **tracked files over 1 MB**, which must stay
> at zero once B-054 closes.
| Commit-time gate | **9 hooks, all proven to fire** ✅ (was: none) | enforced | `pre-commit run` |
