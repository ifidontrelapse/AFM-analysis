# CURRENT TASK

**ID:** `M1-T02`
**Title:** Declare dev dependencies — `pytest`, `pytest-cov`, `ruff`, `mypy`
**Milestone:** M1 — Repository hygiene & quality gates
**Status:** selected — not started
**Branch to use:** `chore/dev-dependencies`
**Estimated size:** S
**Risk to scientific output:** none — no functional code is touched
**Selected:** 2026-08-03

---

## Why this task is next

The project has **no quality gate at all.**

`PROJECT_CONTEXT.md` §14 presents `pytest` and `ruff check .` as "configured checks", and
`pyproject.toml` carries a 40-line `[tool.ruff]` block. Neither tool is declared as a
dependency, and the audit verified that neither is installed:

```
pytest MISSING · ruff MISSING · mypy MISSING
```

So the ruff configuration has never been executed, the `T20` rule that would catch the 13
`print` calls in library code has never run, and the one test file has never been checked
by anything. The only working verification in the repository today is
`tests/characterization/capture.py`, which is invoked by hand.

Every remaining M1 task — repair ruff config, wire the golden into pytest, replace the
fake test, pre-commit, CI — presupposes that these tools exist. This is the unblocking
task.

Reference: `docs/audit/2026-07-28-baseline-audit.md` — defect **D-20**.

---

## Scope

**In scope**

1. Add a dev dependency group to `pyproject.toml` (uv's `[dependency-groups]`):
   - `pytest`
   - `pytest-cov`
   - `ruff`
   - `mypy`
2. `uv sync` and confirm all four resolve and run
3. Record the installed versions in `docs/Development.md`
4. Run each tool once and **record the raw baseline output** — do not fix anything yet:
   - `ruff check .` — expect a large number of findings (T20 prints, import order,
     naming, the Russian-string modules); the count is the M2 starting point
   - `mypy src` — expect many errors; this is why M1-T04 sets baseline exclusions
   - `pytest` — expect the assertion-free `test_io.py` to pass vacuously
5. Do **not** change any rule configuration — that is M1-T03 (ruff) and M1-T04 (mypy)

**Out of scope**

- Fixing any lint or type finding (M2)
- Repairing the ruff configuration keys (M1-T03)
- Writing the mypy configuration (M1-T04)
- Touching anything under `src/`

---

## Definition of done

- [ ] `uv run pytest --version`, `uv run ruff --version`, `uv run mypy --version` all succeed
- [ ] `pyproject.toml` declares the dev group; `uv.lock` updated in the same commit
- [ ] Baseline counts recorded in `docs/Progress.md`: N ruff findings, M mypy errors,
      K tests collected
- [ ] Runtime dependencies unchanged — `git diff pyproject.toml` touches only the new group
- [ ] `python tests/characterization/capture.py` still reports zero drift
- [ ] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [ ] Commit: `M1-T02: declare pytest, ruff and mypy as dev dependencies`

---

## Plan

1. Branch `chore/dev-dependencies`
2. Add `[dependency-groups] dev = [...]` to `pyproject.toml`
3. `uv sync`
4. Run each tool, capture the raw output into the scratchpad, count findings by rule
5. Record the baseline in `Progress.md` — these numbers are the M2 burn-down target
6. Verify the checklist, commit, advance `CURRENT_TASK.md` to `M1-T03`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `uv sync` re-resolves and moves runtime versions, changing golden numbers | Golden `_meta` pins NumPy 2.4.4 / SciPy 1.17.1 / scikit-image 0.26.0. Run `capture.py` after syncing; if a number moves, it is a dependency-driven change and gets its own commit with the version bump recorded (ADR-0008 §5). |
| The torch CUDA 11.8 index re-downloads gigabytes | Adding a dev group should not touch the torch resolution; check `uv.lock` diff before committing. |
| Temptation to fix the flood of ruff findings immediately | Explicitly out of scope. The baseline count is the deliverable; the fixes are M2 work with the golden as a safety net. |

---

## Notes for the next session

After T02 → T03 (ruff config) → T05 (golden into pytest) → T06 (real I/O test) →
T07/T08 (pre-commit, CI). At that point `make check` is meaningful and M2 can begin.

**Do not start M2 before M1-T05 is green.** The characterization harness is the only
thing standing between a large refactor and silent scientific drift.
