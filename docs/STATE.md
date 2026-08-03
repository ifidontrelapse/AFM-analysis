# STATE

**Last updated:** 2026-08-03 · **Branch:** `chore/dev-dependencies` · **Base commit:** `11e0ecc`

> This file is mandatory and must be updated at the end of **every** development session.
> Read it first when a session starts.

---

## Current milestone

**M1 — Repository hygiene & quality gates**

Make the repository reviewable and verifiable before anything else is built.
Exit criteria in `docs/Roadmap.md`.

## Current task

**`M1-T03` — Repair the ruff configuration**

Full detail in `docs/CURRENT_TASK.md`. Status: **selected, not started**.

---

## Completed

### M1 — Repository hygiene (in progress)

- **M1-T01** ✅ (2026-08-03) — tracked files 2 877 → **77**; `frontend/node_modules`
  (2 800 files) untracked; `yolov8s-world.pt` (26 MB) removed from the index before it
  entered history; `.gitignore` rewritten; `.claude/settings.json` now shared; junk
  deleted (`.zip`, four `__pycache__/`, tool caches, empty `output/` and `notebooks/`,
  stray root `package-lock.json`); `plan.md` archived to `docs/archive/`.
  Characterization: **zero drift**.
- **M1-T11** ✅ — absorbed into M1-T01.
- **M1-T02** ✅ (2026-08-03) — pytest 9.1.1, pytest-cov 7.1.0, ruff 0.16.1, mypy 2.3.0
  declared and installed; no runtime version moved, golden still stable. Baseline
  measured: **196 ruff findings** (108 in `src/`), **30 mypy errors**, **1 test, failing**.
  Nothing fixed — that is M1-T03/T04 and M2.

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

Nothing. M1-T02 closed; M1-T03 selected and awaiting execution.

---

## Blocked / needs decision

Decisions only the operator can make. Each blocks a specific task.

| # | Question | Blocks | Why it needs the operator |
|---|---|---|---|
| B1 | **Package name.** `nanoscope` is proposed; distribution name is still `afm-analysis`. The project now covers SEM/TEM, so "AFM" is too narrow. | M2-T01 | Naming is a product decision, and renaming later is expensive |
| B2 | **`min_size_nm` semantics (D-04).** `int(5 / 9.77) == 0` disables the noise filter on 90% of your scans. What *should* the minimum particle size mean at coarse pixel scales — a floor of 1 px, a rounded value, or an error? | M3-T02 | It defines what counts as a particle; that is physics, not engineering |
| B3 | **Detection polarity (D-12).** TEM particles are dark on bright; the detector keeps the bright side and finds 0 of 22. Explicit configuration per modality, or auto-detection? | M3-T10 | Determines whether TEM support is a setting or a heuristic |
| B4 | **Opening-radius rounding (D-10).** Half-integer radii produce an even-sized structuring element with no centre pixel, shifting `z_result` by half a pixel. Round up, round to nearest odd, or floor? | M3-T09 | Changes substrate estimation on real data |
| B5 | **Fate of `frontend/`** (React client for a backend that was never written), `preprocess_batch.py` (broken on every file), and the committed notebooks with outputs. Park, archive, or delete? | M1-T09, M2-T13 | Deleting work is the owner's call, not the engineer's |
| B6 | **Real sample data in git.** `data/` holds 628 SPM scans and is ignored. Should one small representative scan be committed as a test fixture? | M3-T16 | Data ownership and repository size |

None of these blocks M1-T01 or the rest of M1.

---

## Next

1. **Execute `M1-T03`** — repair the ruff configuration: deprecated top-level
   `select`/`ignore`, `target-version = py311` on a 3.12 project, template
   `known-first-party = ["your_package_name"]`, and `fix = true`, which makes
   `ruff check .` rewrite source files
2. `M1-T04` — mypy configuration with baseline exclusions for `src/` (30 errors today)
3. `M1-T05` — wire the characterization golden into `pytest`, so the safety net runs
   automatically instead of by discipline
4. `M1-T06` — replace `tests/test_io.py`; the suite is currently red
5. `M1-T07` / `M1-T08` — pre-commit and CI; at that point the gate is real
6. Answer **B1** so that M2 can start on schedule

---

## Health indicators

| Indicator | Value | Target | Source |
|---|---|---|---|
| Tracked files | **77** ✅ (was 2 854) | < 100 | `git ls-files \| wc -l` |
| Tracked model weights | **0** ✅ (was 1) | 0 | `git ls-files '*.pt'` |
| `.git` size | 81 MB | — | `du -sh .git` — history unchanged, see B-040 |
| Library LOC | 2 021 | — | `wc -l src/**/*.py` |
| Meaningful tests | 0 (1 test, failing) | ≥ 80% of core | `pytest -q` |
| ruff findings in `src/` | 108 | 0 | `ruff check src/ --no-fix` |
| mypy errors | 30 in 9 files | 0 | `mypy src` |
| Characterization phantoms | 8 | 8 | `tests/characterization/` |
| Open audit defects | 24 (5 critical) | 0 critical | audit §2 |
| Import cycles | 5 | 0 | audit D-18 |
| `print` calls in library code | 13 | 0 | audit D-23 |
| Non-English lines in library code | 197 | 0 | audit D-22 |
| Lint/type/test gate | tools installed, all red | green in CI | M1-T03…T08 |
