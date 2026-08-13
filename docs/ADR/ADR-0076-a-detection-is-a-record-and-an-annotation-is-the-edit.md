# ADR-0076 — A detection is a record; an annotation is the edit

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** operator + agent (M7-T07)
- **Affects:** `gui/viewmodels`, `gui/panels` · M7 · M8

## Context

The task list calls this *"manual add/edit/delete of detections"*, and reading it properly is most of
the work. Three things M4 built are still waiting for a caller:
`AnnotationSource.FROM_DETECTION` has never been written by anything, and `UpdateAnnotation` and
`RemoveAnnotation` have no callers outside their own tests.

## Decision

### 1. A detection is not edited

A stored detection is **what a detector produced in a run** (ADR-0042) — a record of something that
happened. An operator deleting one makes the run describe an analysis that never ran, and the project
loses its only honest answer to *"what did the detector actually find?"* — which is the question
M3-T15's evaluation harness exists to ask, and the number M8 will train against.

### 2. Correcting the machine is adopting its answer, and the adoption is marked

ADR-0044 built the way out and said why: an annotation carries `source`, `manual` or
`from_detection`, **because "a model trained on its own output is confirming itself"**. Adopting a
detection creates an annotation with the detector's box and that mark, for ever.

A blob detection has no `bbox` (ADR-0031), so the circle becomes the square that bounds it — a
**stated conversion**, and ADR-0044's own words: *"a circle converts to a box losslessly for
training."*

### 3. Adoption is one click, and adopting everything is one more

Reviewing forty detections means keeping thirty-eight and fixing two. A workflow that costs a dialog
per particle is one nobody uses twice, and both paths go through the command stack, so both undo.

### 4. Deleting one box asks nothing

M5-T04's confirmation exists because removing an **image** destroys hand work that cannot be
recomputed. Deleting **one box** the operator is looking at, with `Ctrl+Z` in the same menu, is the
opposite case — and a dialog there is the one nobody reads by the third time, which is the argument
ADR-0055 made for making confirmations rare.

### 5. Editing acts on a selection the canvas already had

M6-T05 put particle selection in the viewmodel; annotations get the same treatment. A click reaches
the annotation layer first because M7-T01 drew it on top **for exactly this**.

## Consequences

**Positive** — the machine can be corrected without the project forgetting what it said; the
provenance M8 depends on is written by the only workflow that produces it; three pieces of M4 finally
have callers.

**Negative** — an adopted box is a copy: correcting a detector's box means editing the annotation,
and the detection keeps its original coordinates. That is the point, and it means a project carries
both, which somebody comparing them for the first time has to be told.

**Neutral** — editing a polygon's vertices or a painted mask is not offered. The shapes are stored;
a vertex editor is a tool of its own, and nothing has asked for one.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Edit detections in place | The run stops describing what ran; M3-T15's question loses its answer |
| Delete a detection from a run | Same, and the measurement table beside it would disagree |
| Adopt without marking the source | Erases the distinction M8 needs to avoid training on its own output |
| Confirm every deletion | The dialog nobody reads by the third time (ADR-0055) |
| A separate "corrections" table | A third shape for what an annotation already is |

## Compliance

`tests/gui/test_detection_editing.py` asserts an adopted detection is `from_detection` with the
detector's box, that **the stored detections are untouched** after an adoption and a deletion, that
adopting everything is one call, and that rename and delete go through the stack with undo restoring
the same row. The selection tests cover the click, the thicker outline, and the selection clearing
when its box is deleted.

## References

- ADR-0044 — `source`, and why it exists
- ADR-0042 — a run as a record
- ADR-0055 — when a confirmation is worth having
- ADR-0070 / M6-T05 — the layer and the selection this uses
