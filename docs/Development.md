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

Everything below must pass before a merge (PROJECT_RULES §6):

```bash
ruff check .                                   # lint
ruff format --check .                          # formatting
mypy nanoscope                                 # types            (M2)
pytest                                         # tests            (M1-T02 installs it)
python tests/characterization/capture.py       # numerical drift
```

After M1-T10 this is one command:

```bash
make check
```

**Current reality:** `pytest`, `ruff` and `mypy` are declared nowhere and installed
nowhere (audit D-20). M1-T02 and M1-T03 fix this. The characterization runner works
today and is the only gate that currently exists.

---

## 5. The characterization safety net

This is the most important tool in the repository. Read
`docs/audit/characterization-baseline.md` before touching numerical code.

```bash
python tests/characterization/capture.py            # compare — exit 1 on drift
python tests/characterization/capture.py --write    # re-baseline after a declared change
```

- 8 seeded phantoms, ~100 s, CPU only, no weights, no network, no file I/O
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
frontend/               React client, parked — see ADR-0007
notebooks/, *.ipynb     experiments, not interfaces
checkpoints/ data/ dataset/   local only, never committed
```

---

## 9. Things that will bite you

| Trap | Reality |
|---|---|
| `import src.types` looks cheap | It loads 1179 modules including matplotlib and pandas — five import cycles via `src/__init__.py` (D-18). Fixed in M2-T09. |
| `build_substrate_map(manual_radius_px=...)` | Raises `UnboundLocalError` 100% of the time (D-01). Fixed in M3-T01. |
| `preprocess_batch.py` | Fails on every file since `e8caf25` and reports it as `0 converted, N failed` (D-02). |
| `README.md` | Stale: wrong return convention, modules that no longer exist (D-24). Rewritten in M9-T01. |
| YOLO results | Input preparation currently keeps ~12.6% of the dynamic range (D-03). Any YOLO benchmark before M3-T03 is meaningless. |
| Coarse scans | On 90% of real scans the minimum-size filter is silently disabled (D-04). |
| TEM data | The detector keeps the bright side of the Otsu threshold; on dark-on-bright TEM it finds nothing (D-12). |
| Upgrading scikit-image or SciPy | May legitimately move golden numbers. Re-baseline in its own commit, with the version bump, and record the delta. |
