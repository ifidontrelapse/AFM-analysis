# Progress

Append-only session log. Newest entry on top. Every session adds one entry, finished or not.

**Entry format:** date · milestone · task IDs · what changed · what was learned · what is next.
A session that changes scientific output states the numerical delta explicitly.

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
