# CURRENT TASK

**ID:** `M2-T09` · `M2-T10`
**Title:** Break the import cycles; one owned capability matrix
**Milestone:** M2 — Domain extraction (behaviour-preserving)
**Status:** **done 2026-08-04** — both. Rewritten for `M2-T11` at the start of the next
session; the record is in `docs/Progress.md` and `docs/TASKS.md`.
session; the record is in `docs/Progress.md` and `docs/TASKS.md`.

> **M2-T08 was narrowed:** one port, not seven. The reasoning is in
> `nanoscope/core/ports/__init__.py` and in `Progress.md` — six of them have no
> implementation and no caller, and each now has a named task that brings it with its
> first adapter. If that call is wrong, the fix is to say so and write them; nothing else
> in the milestone depends on the outcome.
**Branch to use:** `feat/import-graph-and-capabilities`
**Estimated size:** M
**Risk to scientific output:** **the golden cannot see most of it.** YOLO and SAM2 are not
in the characterization baseline — model inference is excluded from the gate by
PROJECT_RULES §6 as insufficiently reproducible. One exception: `capture.py` records
`yolo_input_preparation`, the deterministic image preparation before inference, for all 8
phantoms. That part is covered; the rest is not
**Selected:** 2026-08-04

> The previous task file described the finished `M2-T04…T06` batch. Its record lives in
> `docs/Progress.md` (2026-08-04, "Three moves, one branch, zero drift") and
> `docs/TASKS.md`.

---

## Why this task is next

Six of the twelve `src/` modules are now shims. What is left is the code that cannot go
into `core/` at all: `yolo_detector.py` and `segmentation.py` import torch, ultralytics,
`patched_yolo_infer` and SAM2. `core` is defined by not importing them (ADR-0001), so this
is the move that makes the dependency rule true rather than aspirational — and M2-T09's
import-graph test can only be written once it is.

---

## Scope

**In scope**

1. `src/detection/yolo_detector.py` → `nanoscope/infrastructure/models/yolo.py`
2. `src/segmentation.py` → `nanoscope/infrastructure/models/sam2.py`
3. Shims left behind in both places, as in M2-T04…T06
4. The heavy imports stay **function-local**. This is not a style question: CI installs
   no torch (M1-T08), and every import in `src/` was verified function-local before that
   environment was chosen. A module-level `import torch` would turn CI red — which is the
   good outcome; a module-level import that CI somehow tolerates would be worse

**Out of scope**

- Defining the `Detector` and `Segmenter` ports — M2-T08, wholesale
- `DeviceManager` / device selection — M4-T12
- The 6 mypy errors in `yolo_detector.py`. Note that mypy sees **more** of them locally
  than in CI, because `ultralytics` is installed here and absent there (M1-T08)

---

## Definition of done

- [ ] Both modules under `nanoscope/infrastructure/models/`, shims in `src/`
- [ ] AST comparison before the gate: every definition code-identical, or every difference
      named — the standard set in M2-T03 and held since
- [ ] **No new module-level heavy import.** Verified the way M1-T08 verified it: import the
      package in the CI environment and assert torch is absent
- [ ] `make check` green; golden zero drift, including `yolo_input_preparation`
- [ ] CI green — and this is the run that matters, since CI is the environment without torch
- [ ] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [ ] Commit: `M2-T07: move the model-backed code to infrastructure`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| A heavy import escapes to module level during the move and CI goes red | That is the mitigation working. The failure to avoid is the opposite — a `try: import torch` that hides the problem. Do not add one. |
| `segmentation.py` imports `src.measure` inside a function (line 57) and would create a new cross-package edge | It becomes `nanoscope.core.science.measurement`, which is infrastructure → core, the correct direction. Confirm it stays that way; a core → infrastructure edge is the thing M2-T09 will fail on. |
| The golden covers almost none of this | Say so rather than implying the green means more than it does. `yolo_input_preparation` is the only covered part, and it is covered for all 8 phantoms. |

---

## Notes for the next session

After M2-T07, `src/` holds `pipeline.py`, `preprocessing_pipeline.py`, `visualization.py`,
`__init__.py` and six shims. Then **M2-T08** (ports) and **M2-T09** (break the five import
cycles, add the import-graph test) are the two that turn the layout into an enforced rule.

Carried, still not tasks: **B-058** (the golden compares CPython exception text — ADR before
any Python upgrade), **B-054** (two README figures over 1 MB, M9-T01).
