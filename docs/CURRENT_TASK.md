# CURRENT TASK

**ID:** `M1-T04`
**Title:** Add the mypy configuration
**Milestone:** M1 — Repository hygiene & quality gates
**Status:** selected — not started
**Branch to use:** `chore/mypy-config`
**Estimated size:** S
**Risk to scientific output:** none — configuration only, no source edits
**Selected:** 2026-08-03

---

## Why this task is next

mypy is installed (2.3.0) and has never been configured. Run with defaults it reports
**30 errors in 9 files**, and one of them is a real defect the audit found by execution:

```
src/pipeline.py:110: error: Argument 4 to "run_sam2_from_blobs" has incompatible type
"ndarray[...] | None"; expected "ndarray[...]"  [arg-type]
```

That is the SEM/TEM path, where `z_flat` is `None` by construction (`pipeline.py:53`).
A configured type checker turns that class of defect from "discovered by running it on
real data" into "caught before commit" — which is the entire argument for M3 and M4 being
survivable.

The configuration question is not "how do we get to zero errors". It is **what posture
lets M2 proceed without either lying about the state of `src/` or blocking on it.**

---

## Scope

**In scope**

1. Add `[tool.mypy]` to `pyproject.toml`:
   - `python_version = "3.12"`, `files`, `pretty`, `show_error_codes`
   - **Strict for new code.** The `nanoscope` package created in M2-T01 is checked
     strictly from its first line — that is far cheaper than retrofitting.
   - **Baseline posture for `src/`.** It carries 30 errors and is scheduled for deletion
     in M2-T15. Options, to be decided in this task and recorded here:
     - a per-module `[[tool.mypy.overrides]]` block relaxing `src.*`, or
     - keep `src/` checked but non-blocking in CI until M2 lands
   - `ignore_missing_imports` for the untyped third-party stack (8 of the 30 errors are
     `import-untyped`: ultralytics, sam2, patched_yolo_infer, cv2, …) — scoped per module,
     never globally
2. Confirm `mypy` runs clean **on an empty strict scope**, so the gate is green from the
   moment M2 creates the package
3. Record the resulting error count and the exact posture in `Progress.md`
4. Classify the 30 errors into *real defects* vs *missing annotations*, and file the real
   ones as M3 tasks or backlog items — a type error that describes a genuine bug must not
   be silenced by configuration

**Out of scope**

- Fixing any type error or adding annotations to `src/` — that is M2 work
- `mypy --strict` over `src/` today; it would produce a wall of noise about a package
  scheduled for deletion
- Wiring mypy into CI (M1-T08)

---

## Definition of done

- [ ] `mypy` runs with an explicit configuration and no command-line flags
- [ ] The strict scope is defined and passes (vacuously today — nothing in it yet)
- [ ] `src/` posture chosen and justified in one paragraph in `Progress.md`
- [ ] The 30 errors classified: N real defects (filed as tasks/backlog), M annotation gaps
- [ ] `ignore_missing_imports` applied per module, never as a blanket setting
- [ ] `python tests/characterization/capture.py` reports zero drift
- [ ] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [ ] Commit: `M1-T04: add the mypy configuration`

---

## Plan

1. Branch `chore/mypy-config`
2. Capture the current 30 errors with codes and file locations into the scratchpad
3. Read each one; separate genuine contract violations from missing annotations
4. Write `[tool.mypy]` plus per-module overrides
5. Re-run; confirm the count is what the configuration says it should be
6. File the real defects; update docs; commit; advance `CURRENT_TASK.md` to `M1-T05`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Configuration silences a real bug.** `ignore_missing_imports` and module overrides make errors disappear; one of the 30 is a genuine `None` contract violation. | Classify all 30 *before* writing the config. Anything real becomes a task with an ID, not a suppressed line. |
| Strictness chosen for `src/` blocks M2 | `src/` is deleted in M2-T15. The configuration must describe a package that is on its way out, not enforce a standard on it. |
| `ignore_missing_imports = true` globally | Hides typos in first-party imports too. Scope it per third-party module. |

---

## Notes for the next session

After T04 → T05 (golden into pytest) → T06 (real I/O test) → T07/T08 (pre-commit, CI).
At that point `make check` is meaningful and M2 can begin.

**Do not start M2 before M1-T05 is green.** The characterization harness is the only
thing standing between a large refactor and silent scientific drift.
