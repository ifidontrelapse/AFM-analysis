# STATE

**Last updated:** 2026-08-04 · **Branch:** `chore/ci` · **Base commit:** `11e0ecc`

> This file is mandatory and must be updated at the end of **every** development session.
> Read it first when a session starts.

---

## Current milestone

**M1 — Repository hygiene & quality gates**

Make the repository reviewable and verifiable before anything else is built.
Exit criteria in `docs/Roadmap.md`.

## Current task

**`M1-T10` — Add a one-command gate**

Full detail in `docs/CURRENT_TASK.md`. Status: **selected, not started**. It is the last
task in M1.

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

Nothing. M1-T09 closed; **M1-T10 is the last task in M1**.

**Two things to know about the repository state:**

- `chore/ci` is pushed to `origin`; `main` is fast-forwarded **locally only** and has not
  been pushed
- The green run was confirmed from the workflow badge. The anonymous GitHub API rate limit
  was exhausted by polling, so per-step timings were not read

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

1. **Push `chore/ci` to `origin`** so the workflow actually runs once — it is written and
   locally verified, but no GitHub run exists. Operator's call
2. **Execute `M1-T09`** — notebooks; `pre-commit run --all-files` is still red on the two
   committed ones, which is M1-T09's property, not a bug in the hooks
3. Answer **B1** so that M2 can start on schedule — it is now the only thing blocking it

---

## Health indicators

| Indicator | Value | Target | Source |
|---|---|---|---|
| Tracked files | **78** ✅ (was 2 854) | < 100 | `git ls-files \| wc -l` |
| Tracked working tree | **7.8 MB** ✅ (was 17 MB) | — | `git ls-files -z \| xargs -0 du -ch` |
| Tracked model weights | **0** ✅ (was 1) | 0 | `git ls-files '*.pt'` |
| `.git` size | 81 MB | — | `du -sh .git` — history unchanged, see B-040 |
| Library LOC | 2 021 | — | `wc -l src/**/*.py` |
| Meaningful tests | **23, all passing** ✅ (was 1, failing) | ≥ 80% of core | `pytest -q` |
| Golden enforced automatically | **yes** ✅ (was: by discipline) | yes | `pytest` |
| `src/` modules with a unit test | 1 of 12 (`afm_io`) | 12 | `tests/unit/` |
| ruff findings, legacy core | 117 (109 `src/` + 8 `preprocess_batch.py`) | 0 | `ruff check src preprocess_batch.py --no-fix --no-force-exclude` |
| ruff findings, code we own | **0** ✅ | 0 | `ruff check . --no-fix` |
| mypy errors | 22 in 7 files locally, **21 in CI** (no `ultralytics` → less inference) | 0 | `mypy` |
| Characterization phantoms | 8 | 8 | `tests/characterization/` |
| Open defects | 28 (24 audit + 3 mypy + 1 found by the M1-T06 tests) | 0 critical | audit §2, M3-T17…T20 |
| Import cycles | 5 | 0 | audit D-18 |
| `print` calls in library code | 13 | 0 | audit D-23 |
| Non-English lines in library code | 197 | 0 | audit D-22 |
| Lint/type/test gate | **green end to end** ✅ — hooks on commit, CI on push | stays green | GitHub Actions |
| Commit-time gate | **9 hooks, all proven to fire** ✅ (was: none) | enforced | `pre-commit run` |
