# CURRENT TASK

**ID:** `M6-T04`
**Title:** The predictor the matrix keeps asking for, and the masks it produces
**Milestone:** M6 — Analysis workflow in the GUI, fourth task
**Defect:** — · **ADR:** **ADR-0064**
**Branch:** `feat/m6-analysis-workflow`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M6-T02 disables every `segment` row with *"segmentation needs a loaded predictor, which arrives in
M6-T04"*. This is M6-T04, and the sentence is a promise with a date on it.

It is also the last unwired half of ADR-0050: the registry hands back **factories**, and until now
nothing has ever called one. `resolve()` has no caller outside its own tests.

---

## The decisions this task has to make

**1. Who builds the predictor?** The composition root, lazily, once.

`Nanoscope.segmentation_predictor()` looks up a `SEGMENT`-task model in the open project, resolves
its factory (ADR-0050), and constructs it with the device `select_device` chose (ADR-0049). It is
`app/`'s work by PROJECT_RULES §2.7 — nothing else constructs infrastructure — and it is **cached**,
because building it loads weights off a disk.

**2. When?** Inside the job, never on the main thread.

Loading weights is seconds of I/O and GPU allocation. `has_predictor` for the *matrix* is answered by
a **registered model**, not a constructed one: ADR-0050 made the registry cheap on purpose, so
asking "can this project segment?" must not load anything.

**3. Where does the panel for it go?** There is no new panel, and that is the decision.

The detection panel already offers modes from the matrix, and `segment` is one of its rows. A second
panel would have its own detector and mode choices — the duplication ADR-0062 exists to prevent, one
task after it was written. What this task adds is the mode becoming *available*, and the masks
becoming visible.

**4. The mask parameters are not offered, and the reason is worth writing down.**
`PipelineConfig`'s fields for them are named after the framework, **the golden records that class's
field names**, and a widget setting them would have to write the name PROJECT_RULES §2.5 forbids in
`gui/`. Renaming the fields is a golden-moving change and gets its own commit (ADR-0010). They stay
at their defaults, which is what every call before this one used.

**5. How does a mask reach the screen if masks are not stored?**

ADR-0042 did not persist them — SAM2's weights are outside the gate, so the format would have been
written blind. So the run **carries them in memory**: `AnalysisRun.masks`, empty on everything the
repository returns, filled on the run you just computed. The overlay therefore shows masks for this
session's run and says nothing about older ones, which is the truth about what is stored.

**6. What is drawn?** The mask's outline, not a filled sheet.

A filled overlay hides the pixels it describes, and the pixels are the measurement. One outline per
mask, in the same accent as the detections, under the same kind of toggle.

---

## Scope

**In scope**

1. `app/container.py` — `segmentation_predictor()`: registry → factory → device, cached
2. `core/entities/project.py` — `AnalysisRun.masks`, in memory only
3. `application/use_cases/analysis.py` — the masks travel back with the run
4. `gui/viewmodels/session.py` — a predictor when the mode needs one, `has_predictor` from the
   registry
5. `gui/panels/viewer.py` — the mask outlines and their toggle
6. **ADR-0064** — who builds the predictor, why there is no second panel, and why masks are
   in-memory
7. Tests, with a **stub predictor** registered through the registry — the only way this path can be
   tested at all, and M3-T14's precedent

**Out of scope**

- **Persisting masks** — a format decision with a migration behind it; ADR-0042's deferral stands
- **The mask parameters** — decision 4
- **Choosing between two segmentation models** — a project has at most one today

---

## Definition of done

- [x] A registered segmentation model makes the mode selectable, without loading weights
- [x] The predictor is built once, in the job, by the composition root
- [x] Masks from the run just computed are drawn as outlines, and can be turned off
- [x] ADR-0064 + the ADR index
- [x] `make check` green — 1095 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M6-T04: the predictor the matrix keeps asking for, and the masks it produces`

---

## What it turned up

**The panel this task was scheduled to build already existed.** The detection panel offers modes
from the matrix, and `segment` is one of its rows — a segmentation panel would have carried a second
detector and mode choice, which is precisely what ADR-0062 was written to prevent one task earlier.
Closed with an argument rather than with code, the way M4 closed three of its own.

**A `has_predictor=False` written in M6-T02 survived the edit that was supposed to replace it** —
the formatter had reflowed the block my replacement was matching on, so the substitution silently
did nothing and the mode stayed disabled with a registered model in the project. Caught by the test
that asserted the opposite. **A search-and-replace that matches nothing is a change that did not
happen**, and only the test noticed.

**The refusal sentence named a task that had just shipped.** *"...which arrives in M6-T04"* would
have been wrong the moment this commit landed; it now says what is actually missing — a registered
model — which is a sentence that stays true.
