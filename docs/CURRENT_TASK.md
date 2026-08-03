# CURRENT TASK

**ID:** `M1-T05`
**Title:** Wire the characterization harness into pytest
**Milestone:** M1 — Repository hygiene & quality gates
**Status:** selected — not started
**Branch to use:** `chore/golden-in-pytest`
**Estimated size:** S–M
**Risk to scientific output:** none — the harness is wrapped, not modified
**Selected:** 2026-08-04

---

## Why this task is next

This is the gate item that actually matters.

`tests/characterization/capture.py` is the only check in the repository that passes, and
it is the **only protection M2 has**. M2 moves every scientific module in the project into
a new package; the guarantee that the move changed nothing rests entirely on this
comparison reporting zero drift.

Today it runs when someone remembers to type the command. That is not a safety net, it is
a habit — and habits do not survive a 16-task refactor, a tired evening, or an agent that
was not told about it. ADR-0008 makes zero drift a merge requirement; this task makes the
requirement mechanically checkable.

**M2 must not start until this is green.** That is stated in three places already, and
this is the task that lifts the block.

---

## Scope

**In scope**

1. Add `tests/characterization/test_golden.py` — a real pytest test that invokes the
   existing comparison and fails with the path-addressed diff as the assertion message
2. Do **not** modify `capture.py`'s logic. Import and call it; if its current structure
   makes that awkward, add a thin callable entry point, keeping the CLI behaviour identical
3. Mark it `@pytest.mark.slow` (~100 s) and register the marker in configuration so it
   does not warn
4. Define how the fast and full runs are selected: `pytest` runs everything by default;
   `pytest -m "not slow"` skips it for inner-loop work. Document both.
5. Make the failure output useful — a drift report must name the quantity that moved and
   its before/after values, exactly as the CLI does today
6. Verify the test **fails** when it should: temporarily perturb one golden value, confirm
   a red run and a readable message, then restore. This is the only proof that the safety
   net is connected.
7. `pytest.ini` currently holds only the `pythonpath` hack. Decide: fold pytest
   configuration into `pyproject.toml` now, or leave `pytest.ini` until M2-T14 deletes the
   hack. Record the choice.

**Out of scope**

- Changing any golden value
- Fixing `tests/test_io.py` (M1-T06) — the suite stays red until then, and that is expected
- Coverage reporting (pytest-cov is installed but unwired; a separate small task)
- CI (M1-T08)

---

## Definition of done

- [ ] `pytest` runs the golden comparison and it passes
- [ ] `pytest -m "not slow"` skips it and completes in under a second
- [ ] A deliberately perturbed golden makes the test **fail**, with the moved quantity
      named in the output — verified, then reverted
- [ ] No marker warnings; no changes to `capture.py`'s numerical behaviour
- [ ] `python tests/characterization/capture.py` still works as a CLI, unchanged
- [ ] `docs/Development.md` documents both invocations
- [ ] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [ ] Commit: `M1-T05: run the characterization golden under pytest`

---

## Plan

1. Branch `chore/golden-in-pytest`
2. Read `capture.py` to find the cleanest seam between "compare" and "print/exit"
3. Write the test; register the `slow` marker
4. Prove the negative case by perturbing a golden value; revert
5. Update `Development.md` and the docs; commit; advance `CURRENT_TASK.md` to `M1-T06`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Refactoring `capture.py` to make it testable changes what it measures** | Add a callable entry point around the existing logic; do not touch comparison or tolerance. Run the CLI before and after and diff the output. |
| A 100 s test makes people stop running `pytest` | `-m "not slow"` for the inner loop; the full run is the merge gate. Document both prominently. |
| The test passes because it silently skips | This is the failure mode that would quietly remove the safety net. Hence the mandatory perturbation check — a test that cannot fail is not a test. |
| `--write` becomes an easy escape from a red run | ADR-0008: re-baselining without an ADR is a rule violation. M1-T07 pre-commit and M1-T08 CI make an undeclared golden change visible in review. |

---

## Notes for the next session

After T05 the block on M2 lifts. Remaining M1: T06 (real I/O test), T07 (pre-commit),
T08 (CI), T09 (notebooks), T10 (`make check`).

**B1 is still unanswered** — the package name blocks M2-T01, and every M2 task after it
depends on that name. Worth resolving before T05 finishes.
