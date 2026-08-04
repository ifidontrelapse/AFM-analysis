# CURRENT TASK

**ID:** `M1-T08`
**Title:** Add CI
**Milestone:** M1 — Repository hygiene & quality gates
**Status:** selected — not started
**Branch to use:** `chore/ci`
**Estimated size:** M
**Risk to scientific output:** none — CI observes, it does not edit
**Selected:** 2026-08-04

---

## Why this task is next

Pre-commit (M1-T07) deliberately refuses to run the slow half of the gate: `pytest` with
the 200 s golden, and the lint/type findings on `src/`. Those checks exist, they pass, and
right now nothing runs them except a person who remembers to. That is the same gap M1-T05
closed for the golden, one level up.

CI is also what makes the M1-T07 posture honest. The hooks skip `src/` on purpose — 109
ruff findings and 22 mypy errors would block every commit — but "reported, not silenced"
only means something if something reports. Today nothing does.

And it is the last piece before M2. Sixteen relocation tasks, each of which must prove
zero golden drift, on a machine that is not the author's.

---

## Scope

**In scope**

1. `.github/workflows/ci.yml` — GitHub Actions, Ubuntu, Python 3.12, `uv` for the
   environment, triggered on push and pull request
2. Job order, fail-fast on the cheap things: `ruff format --check` → `ruff check` →
   `mypy` → `pytest` (golden included)
3. **CPU-only, no weights, no network.** No CUDA index resolution, no SAM2 checkout, no
   model download. The test suite already respects this; the dependency install must too —
   see the risk below, this is the hard part of the task
4. **Two postures, deliberately different**, mirroring M1-T04 and M1-T07:
   - `tests/` and any new package: **blocking**
   - `src/` and `preprocess_batch.py`: **reported, not blocking** — publish the counts as
     a summary so a regression is visible in review without freezing M2
5. Cache the uv environment; a run that takes fifteen minutes gets ignored like a slow hook
6. Run `pre-commit run --all-files` in CI **only** over the paths the hooks actually own,
   or not at all — decide, and record why. It is currently red on the two committed
   notebooks (M1-T09) and one archived doc
7. A status badge in `README.md` only if it is accurate; `README.md` is stale until M9, so
   a badge that claims health it does not have is worse than none

**Out of scope**

- Fixing the 109 ruff findings or 22 mypy errors (M2/M3)
- Notebook hygiene (M1-T09) — CI must not be what forces it
- Publishing coverage (pytest-cov is installed and unwired; separate task)
- Release/publish workflows, matrix builds across OSes or Python versions

---

## Definition of done

- [ ] A pushed branch produces a green run, from a clean checkout, with no local state
- [ ] The golden runs in CI and reports zero drift there — **the M2 precondition**
- [ ] Total wall time under ~8 minutes, warm cache
- [ ] `src/` findings appear in the run summary without failing the job; a *new* finding in
      `tests/` or a new package **does** fail it
- [ ] No CUDA wheel, no SAM2 clone, no weight download in the install step — verified by
      reading the log, not by assuming
- [ ] A deliberately broken commit (a failing test, then a drifted golden) is **rejected**
      by CI — pushed to a scratch branch, confirmed red, deleted
- [ ] `docs/Development.md`, `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [ ] Commit: `M1-T08: add CI`

---

## Plan

1. Branch `chore/ci`
2. Read `pyproject.toml`'s `[tool.uv.sources]` first — torch resolves from a CUDA 11.8
   index and `sam-2` from GitHub. Decide how CI installs *only* what the tests need
3. Write the workflow; push to a scratch branch and iterate there, not on the task branch
4. Break it on purpose twice — a failing assertion, then a perturbed golden — and confirm
   red both times
5. Record the actual timings, and the size of the `src/` report
6. Update the docs; commit; advance `CURRENT_TASK.md` to `M1-T09`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **`uv sync` in CI drags in torch from the CUDA index and a GitHub checkout of SAM2** — minutes of download for code the tests never import | The suite is CPU-only and imports neither. Install the dev group plus the numerical stack only, or use a CPU torch index for CI. Verify by reading the install log; if a CUDA wheel appears, the job is wrong even when it is green. |
| CI reports the `src/` baseline as a failure and freezes M2 | Non-blocking step for `src/`, blocking for everything else. Same split as `.pre-commit-config.yaml`; keep the two in agreement or they will disagree silently. |
| The 200 s golden makes every run slow, so people stop reading CI | It is the one check M2 depends on — it runs. Keep the rest fast and cached so the golden is the floor, not the addition. |
| A green CI badge on a stale `README.md` implies the project is finished | Badge only if it is accurate about what it measures. README is M9's problem; do not create a second wrong claim to fix later. |
| Hook versions in CI drift from `pyproject.toml` | M1-T07 made `pyproject.toml` the only place a ruff version is declared. CI must call the same `uv run ruff`, not a marketplace action that pins its own. |

---

## Notes for the next session

After T08 the gate is real end to end. Remaining M1: T09 (notebooks), T10 (`make check`).

`pre-commit run --all-files` is knowingly red on the two committed notebooks and one
archived doc. That is M1-T09's property — do not let CI be the thing that forces it, and do
not fix it here.

**B1 is the only thing blocking M2**, open since M0. Every M2 task depends on the package
name. It should be answered before M1 closes.
