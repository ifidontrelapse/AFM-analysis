# Development

How to set up, run, verify and contribute to this project.

> Some commands below describe the **target** state (M1/M2). Where a command does not
> work yet, the task that makes it work is named.

---

## 1. Requirements

| | |
|---|---|
| OS | Linux (primary target) |
| Python | 3.12+ |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| GPU | optional — CUDA, ROCm, MPS or CPU; the `DeviceManager` decides (M4-T12) |
| Qt | PySide6, added in M5 |

---

## 2. Setup

```bash
uv sync
```

PyTorch resolves from the CUDA 11.8 index configured in `pyproject.toml`
(`[tool.uv.sources]`). For CPU-only or a different CUDA version, change that block.

SAM2 installs from its GitHub source. Model weights are **not** in the repository and
never will be — place them under `checkpoints/`:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt \
     -O checkpoints/sam2.1_hiera_base_plus.pt
```

---

## 3. Running

### Today (library)

```python
from src.preprocessing_pipeline import run_preprocessing
from src.pipeline import run_full_pipeline
from src.types import PipelineConfig

pre = run_preprocessing("data/sample.spm", fmt="spm")
res = run_full_pipeline(pre, PipelineConfig(detector="log", mode="baseline"))
print(res.measurements.head())
```

`mode="segment"` additionally requires an initialised `SAM2ImagePredictor` passed as
`predictor=`. The library never loads weights on your behalf.

### After M5 (application)

```bash
nanoscope                 # launch the desktop application
nanoscope --project PATH  # open a project directory
```

---

## 4. Quality gate

One command, and it is what CI runs (M1-T10):

```bash
make check          # ruff format --check → ruff check → pytest, stopping at the first failure
```

`make` on its own lists the targets and runs nothing:

| Target | Command it runs | Blocking |
|---|---|---|
| `check` | `format` → `lint` → `test`, in that order | yes — this is the gate |
| `format` | `ruff format --check .` | yes |
| `lint` | `ruff check . --no-fix` — excludes the legacy core | yes |
| `test` | `pytest` — tests + numerical drift (~200 s) | yes |
| `fast` | `pytest -m "not slow"` (~1 s) | no — the inner loop |
| `golden` | the characterization golden alone | part of `test` |
| `types` | `mypy --no-pretty` — reports on `src/` (M2: `nanoscope`) | **no**, exits 1 today |
| `lint-legacy` | `ruff check src --no-force-exclude --statistics` | **no**, exits 1 today |

`.github/workflows/ci.yml` invokes these targets rather than repeating the commands, so
the local gate and CI cannot describe different things — the M1-T08 near-miss was exactly
that failure, one exclusion declared in two files. The `Makefile` is the single
description of *what the gate runs*; this table only names the targets — if it ever
disagrees with the file, the file is right.

`types` and `lint-legacy` are deliberately outside `check`: they report the legacy `src/`
baseline, which is non-zero by design (M1-T04, M1-T07), so a `check` that included them
could never pass and would be bypassed within a day. CI publishes their output to the run
summary instead.

`ruff check .` skips `src/` (`extend-exclude` + `force-exclude`
in `pyproject.toml`, declared once and shared by the hooks and CI). That is deliberate:
109 open findings there would block every M2 commit. See *CI* below for how to measure
them anyway.

`pytest` now includes the characterization golden (M1-T05), so the drift check is no
longer a separate command you have to remember. For the inner loop, skip it:

```bash
make fast                                      # everything except the golden (~1 s)
```

The `slow` marker is registered in `pyproject.toml`. **`make fast` is for editing, not for
merging** — a merge runs `make check`.

Pytest configuration lives in `[tool.pytest.ini_options]` in `pyproject.toml`. The old
`pytest.ini` was deleted: while it existed, pytest ignored the `pyproject.toml` section
entirely, and two files that can silently override each other is a trap. The
`pythonpath = [".", "src"]` hack survives there until M2-T14 installs the package.

### Pre-commit — the fast half, run automatically

**A fresh clone has no hooks.** Install them once, per clone:

```bash
uv run pre-commit install
```

After that, `git commit` runs ruff format, ruff check, a 1 MB file-size limit,
end-of-file/whitespace fixers, merge-conflict and YAML/TOML checks, and `nbstripout`.
On a normal diff it costs about a second. `pytest` and mypy are deliberately **not** hooks
— the golden alone takes 200 s, and a hook that slow is a hook people bypass. They run in
CI (M1-T08).

Hooks that **rewrite** a file (ruff format, the whitespace fixers) skip `src/`; hooks that
only **refuse** apply everywhere, `src/` included. The
reason is in `.pre-commit-config.yaml`: the core has 109 open ruff findings, so a blocking
hook there would make every M2 commit impossible, and rewriting the science to satisfy a
formatter is PROJECT_RULES §4.1. Same posture mypy takes — reported, not silenced, not
rewritten.

> **`pre-commit run --all-files` edits your working tree, including uncommitted work.**
> It ignores the index and rewrites files on disk. In M1-T07 it silently restored a
> missing final newline in an uncommitted `project.md`. Commit or stash first.

When a hook modifies a file, the commit aborts by design — inspect the change, `git add`
it, commit again. `--no-verify` exists; using it routinely means the hook is wrong, so fix
the hook.

### CI — the slow half

`.github/workflows/ci.yml`, on every push and pull request: format → lint → tests →
legacy report. About four minutes, of which the golden is three. **Each of those steps is
a `make` target** (M1-T10) — the workflow chooses the environment and the order, the
Makefile owns the commands. The steps stay separate rather than one `make check` only so
that a red job names the stage without anyone opening a log.

**CI installs a smaller environment than you have.** `uv sync --only-group ci --locked`
brings numpy, scipy, scikit-image, pandas, matplotlib, headless OpenCV and the dev tools —
and no torch, ultralytics, sam2 or patched-yolo-infer. Every heavy import in `src/` is
function-local, so the suite never touches them, and CI has no reason to resolve the CUDA
11.8 wheel or clone SAM2. A step asserts this rather than trusting it: if a CUDA wheel ever
appears, the job fails even when the tests pass.

Two consequences worth knowing:

- `uv run` re-syncs the project environment unless told not to, which would reinstall
  everything the `ci` group leaves out. The workflow sets `UV_NO_SYNC: "1"` at job level.
- **mypy reports 21 errors in CI and 22 locally.** With `ultralytics` absent, mypy infers
  `Any` where it would otherwise see `list[Any]`, and `yolo_detector.py:99` — part of
  M3-T18 — stops being visible. The type checker's output depends on which third-party
  packages are installed. Neither number is wrong; the local one is more complete.

`src/` is reported, never blocking — the same posture as the hooks and as mypy (M1-T04).
The counts go to the run summary so a regression shows up in review without freezing the
sixteen M2 relocation tasks.

**When the code moves to `nanoscope/`, the checks get sharper but not instantly total**
(M2-T03). `ruff check` becomes *blocking* on moved science, with six named rules ignored
for `nanoscope/core/science/` — Russian text (M2-T12), `print` (M2-T11),
implicit-optional (M3), `RET504`. mypy runs that subtree at its default strictness rather
than the strict `nanoscope.*` settings. Both are declarations that code is in transit, not
exemptions: every entry names the task that deletes it, and everything outside
`core/science/` is strict and at zero. Legacy code cannot satisfy strict rules the same
commit it arrives, and fixing the defects mid-move would change numbers the golden
records — that is M3's job, with a declared delta.

`pre-commit run --all-files` is **not** run in CI. Its blocking content — ruff check and
format — already runs directly with the same configuration, so running it again would only
duplicate the same failures under a second name. (It has been green across the whole
repository since M1-T09, so this is a design choice, not an exemption.)

To measure the legacy baseline yourself — the exclusion has to be overridden explicitly,
which is why it is a target and not something to retype:

```bash
make lint-legacy       # ruff inside src/
make types             # mypy
```

**Tool versions** (installed by M1-T02, declared in `[dependency-groups] dev`):

| Tool | Version | State |
|---|---|---|
| pytest | 9.1.1 | **green** — 23 tests (M1-T05, M1-T06) |
| pytest-cov | 7.1.0 | installed, not yet wired |
| ruff | 0.16.1 | configured (M1-T03); clean outside `src/`, **109 findings** inside it |
| mypy | 2.3.0 | configured (M1-T04); **22 errors** locally / 21 in CI, deliberately not silenced |
| pre-commit | 4.6.1 | configured (M1-T07); `pre-commit install` required per clone |

**Current reality:** the suite is green, the golden runs inside it, hooks refuse on commit
and CI runs the slow half. The lint and type findings that remain are real and confined to
`src/` — they are defects fixed in M2/M3, not configuration noise.

---

## 5. The characterization safety net

This is the most important tool in the repository. Read
`docs/audit/characterization-baseline.md` before touching numerical code.

```bash
pytest tests/characterization/test_golden.py        # the normal way — part of `pytest`
python tests/characterization/capture.py            # same comparison, CLI — exit 1 on drift
python tests/characterization/capture.py --write    # re-baseline after a declared change
```

The test and the CLI share one code path (`capture.diff_against_golden()`), so they cannot
disagree. The CLI stays because `--write` and the standalone exit code are useful outside
pytest.

- 8 seeded phantoms, ~190 s, CPU only, no weights, no network, no file I/O
- Tolerance `rtol=1e-6, atol=1e-9`; counts, dtypes and error types must match exactly
- Drift is reported as a path-addressed diff:
  ```
  afm_flat_monodisperse.log_detection.detect_particles_p20.n_blobs: 24 -> 23
  ```

**The rule.** A refactor must leave every number unchanged. If a number moves, either
the refactor has a bug, or the change was intentional — in which case it needs an ADR,
a test, the regenerated golden **in the same commit**, and a quantified delta in
`Progress.md`.

`--write` without a corresponding ADR is a rule violation.

---

## 6. Workflow

```
1 Read → 2 Analyze → 3 Plan → 4 Document → 5 Implement → 6 Test → 7 Document
```

1. Read `docs/STATE.md`, then `docs/CURRENT_TASK.md`.
2. Branch: `type/short-slug` — `feat/`, `fix/`, `refactor/`, `docs/`, `chore/`, `sci/`.
   `sci/` marks a branch that intentionally changes scientific output.
3. Write the plan into `CURRENT_TASK.md` **before** writing code.
4. Implement the smallest change that satisfies the task. Do not touch unrelated files.
5. Run the gate.
6. Update `STATE.md`, `Progress.md`, `TASKS.md` — and `PROJECT_CONTEXT.md` if `src/`
   changed.
7. Commit: `M1-T01: untrack node_modules and rewrite .gitignore`

One task per branch, one intent per commit.

---

## 7. Adding things

### A new detector or segmenter

1. Implement the `Detector` / `Segmenter` port from `core/ports/`.
2. Put the implementation in `infrastructure/models/<name>/`.
3. Register it in the registry — one line.
4. Add it to the capability matrix in `application/capabilities.py`.
5. Add contract tests against the port; add phantom coverage if it is a pure algorithm.

No other file changes. If you find yourself editing `pipeline`, `gui/`, or a config
dataclass to add a model, the abstraction has failed — say so instead of working around it.

### A new test

Fast tests go in `tests/unit/`. Fixtures are **generated, never committed** — build the
bytes or the array in the test module, seeded (PROJECT_RULES §7,
`tests/characterization/phantoms.py`, `tests/unit/test_afm_io.py`). Nothing may depend on
`data/`, on model weights, or on the network.

Then make the test fail on purpose: break the code it covers, confirm it goes red, revert.
M1-T06 wrote 23 tests this way and found that one of them could not fail — the mutation is
what proved the other 22.

### A new ADR

```bash
cp docs/ADR/TEMPLATE.md docs/ADR/ADR-0012-my-decision.md
```

Number sequentially, never reuse a number, never edit an accepted ADR — supersede it.

### A new task

Add it to `docs/TASKS.md` under its milestone with the next free ID. Ideas without a
milestone go to `docs/Backlog.md`.

---

## 8. Repository layout

```
src/                    scientific core — moves to nanoscope/core in M2
tests/
  unit/                 fast tests, no I/O beyond tmp_path, no fixtures in git
  characterization/     golden safety net (works today)
docs/
  PROJECT_RULES.md      the constitution — read first
  STATE.md              live state — read at session start
  CURRENT_TASK.md       the one task in progress
  TASKS.md              full breakdown
  Roadmap.md            milestones
  Backlog.md            unscheduled
  Progress.md           session log
  Architecture.md       target architecture
  Development.md        this file
  ADR/                  decisions
  audit/                Phase 0 audit — historical, frozen
PROJECT_CONTEXT.md      machine-readable map of the current implementation
notebooks/              experiments, not interfaces — outputs stripped on commit
checkpoints/ data/ dataset/   local only, never committed
```

---

## 9. Things that will bite you

| Trap | Reality |
|---|---|
| `import src.types` looks cheap | It loads 1179 modules including matplotlib and pandas — five import cycles via `src/__init__.py` (D-18). Fixed in M2-T09. |
| `build_substrate_map(manual_radius_px=...)` | Raises `UnboundLocalError` 100% of the time (D-01). Fixed in M3-T01. |
| `README.md` | Stale: wrong return convention, modules that no longer exist (D-24). Rewritten in M9-T01. |
| YOLO results | Input preparation currently keeps ~12.6% of the dynamic range (D-03). Any YOLO benchmark before M3-T03 is meaningless. |
| Coarse scans | On 90% of real scans the minimum-size filter is silently disabled (D-04). |
| TEM data | The detector keeps the bright side of the Otsu threshold; on dark-on-bright TEM it finds nothing (D-12). |
| Upgrading scikit-image or SciPy | May legitimately move golden numbers. Re-baseline in its own commit, with the version bump, and record the delta. |
| **Upgrading Python** | Also moves the golden, and not by changing a number. `capture.py` records exception *messages* verbatim, and CPython rewords them between minor versions — 3.14 turned `too many values to unpack (expected 2)` into `… (expected 2, got 3)`, which reads as characterization drift. CI is pinned to 3.12 and asserts it. See **B-058**. |
