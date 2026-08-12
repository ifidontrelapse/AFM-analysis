# CURRENT TASK

**ID:** `M4-T07`
**Title:** Annotations: the first data the operator makes, not the application
**Milestone:** M4 — Application layer, seventh task
**Defect:** — (W9: annotations cannot become a dataset) · **ADR:** **ADR-0044**
**Branch:** `feat/m4-application-layer` — M4 changes no scientific output (PROJECT_RULES §7)
**Status:** **done 2026-08-12.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

Everything stored so far is either the operator's *file* or the application's *derivation* of it.
An annotation is neither: it is judgement, typed in by hand, and it is the one thing in a project
that cannot be recomputed. W9 names what it is for — *"annotations cannot become a dataset"* — and
M7/M8 cannot start until there is something to train on.

It is also the trigger ADR-0041 named for revisiting import deduplication: two copies of one scan
are untidy until somebody annotates both, and then they are two half-done jobs.

---

## The decisions this task has to make

**1. A table, or JSON files under `annotations/`?** A **table** — and the contrast with M4-T05
matters, because ADR-0042 sent the measurement table to a file two tasks ago.

The rule that separates them is not "files versus database". It is: *does the shape vary, and is it
derived?* A measurement table's columns depend on which producer wrote it (ADR-0031) and can be
recomputed from the image. An annotation is a fixed handful of numbers, edited **one at a time**
with undo behind it (M4-T08), and it is irreplaceable. Rewriting a JSON document per keystroke is
the shape of a lost file; a row per annotation is what a row is for.

ADR-0003's layout mentions "manual annotations (JSON)" — written before ADR-0031 and before undo
was scheduled. `annotations/` keeps its meaning for painted masks, which are files by the same
ADR's rule about bitmaps.

**2. What can an operator draw?** A **box**, with a label. Not a union of point, circle, polygon and
mask.

A box is what a training set consumes, which is the only named consumer (M8), and what a drag
produces. Each additional shape needs storage, an editor, a converter and a test — for a consumer
nobody has written. A circle converts to a box losslessly for training purposes; if M6 finds an
operator drawing something a box cannot express, that is a shape with a reader and this decision
gets revisited then.

**Masks stay deferred** for the third time (ADR-0042 §3): painting is M6, and a format written
before its painter is written blind.

**3. Where did an annotation come from?** A `source` column: `manual` or `from_detection`.

Training a model on boxes copied from that model's own output is self-confirmation, and a training
set that cannot tell the two apart cannot avoid it. One column, two values, a `CHECK` — and the
question it answers is one M8 must ask.

**4. What happens to annotations when their image is removed?** They cascade.

The row they belong to is gone, so keeping them would leave hand-drawn boxes pointing at nothing.
But **`remove_image` is an operator's explicit "forget this scan"**, and it now silently discards
hand work — so the count has to be *askable* before it happens. `annotations_for` is what a
confirmation dialog counts with, and the ADR says so out loud rather than leaving M6 to discover
it.

**5. Coordinates: integers or floats?** Floats. A drag is not on the pixel grid, and rounding at
storage time is a decision the trainer should make with the whole box in hand, not the database.
`Detection.bbox` stays integer — that is a detector's output, and these are two different things
that happen to have four numbers.

A zero-area box is refused by a `CHECK`: it is a mis-drag, and as a training example it is a
picture of nothing.

**6. A use case?** No. `add_annotation` and its siblings are repository calls, and a function that
forwards one call to one object is ADR-0041's case for the fourth time. The use case with policy in
it — adopting a run's detections as a starting point for correction — arrives with the UI action
that calls it (M6), because what it should skip and what it should label is a question about an
editor that does not exist.

---

## Scope

**In scope**

1. Migration step 3: the `annotations` table
2. `Annotation` in `core/entities/project.py`
3. Repository: `add_annotation`, `annotations_for`, `update_annotation`, `remove_annotation`, and
   the port extended to match
4. **ADR-0044** — a table not a document, one shape not a union, provenance, the cascade
5. Tests: round trip, ordering, update, cascade, both `CHECK`s, and a project that keeps its
   annotations across a session

**Out of scope**

- **Painted masks** — decision 2, third deferral
- **Adopting detections as annotations** — decision 6, with M6
- **Undo** — M4-T08, which is what `update_annotation` and `remove_annotation` exist to be undone
- **A class registry.** The label is free text until a dataset needs a vocabulary (M8)
- **Export to a training format** — M8. This task stores what M8 will read

---

## Expected blast radius

- **Zero golden differences.** No numerical code is imported
- One migration step, one table, one entity, one ADR
- No new dependency

---

## Definition of done

- [x] Schema v3 with the `annotations` table and its constraints
- [x] `Annotation`, and four repository methods on the port
- [x] ADR-0044
- [x] Tests including the cascade and both constraints
- [x] `make check` green — 639 tests, golden byte-identical
- [x] `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, `ProjectFormat.md`, ADR index
- [x] Commit: `M4-T07: an annotation is the one thing that cannot be recomputed`

---

## What it turned up

**The migration mechanism refused a database that lied about its version, and that is the finding.**
M4-T05's test fabricates a "v1" database by dropping the tables v2 added; with v3 in the list, that
database still carried an `annotations` table while claiming to be version 1, and step 3 failed on
`CREATE TABLE annotations`. The fix was in the test — drop everything above the target — and the
lesson is that a half-reverted database is not an old database, which is exactly what the mechanism
is supposed to notice.

**The cascade needed a decision, not a default.** Annotations following their image out of the
database is correct, and it is also the first time deleting a row destroys work that cannot be
recreated. Neither refusing the deletion nor adding `force=True` is right — the first puts a UI
decision in storage, the second is a warning nobody reads. What the ADR does instead is hand M6 an
obligation with the tool to meet it.

---

## Notes

The golden held for the seventh time. **M4-T08** takes undo/redo — the reason an edit keeps its id
— and closes another of M4's exit criteria.
