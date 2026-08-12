# ADR-0044 — An annotation is a row, because it cannot be recomputed

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T07)
- **Affects:** `core/entities`, `core/ports`, `infrastructure/storage`, schema v3 · M6 · M8

## Context

Everything a project has stored so far is either the operator's **file** or the application's
**derivation** of it. An annotation is neither: it is judgement, entered by hand, and the only
thing in a project that cannot be produced again by running something.

W9 names what it is for — *"annotations cannot become a dataset"* — and neither M7's training
provider nor M8's dataset export can start until there is something to train on.

Two tasks ago ADR-0042 sent the measurement table *out* of the database and into a file. Deciding
the opposite here needs the rule that distinguishes them, written down, or the next person reads
the two decisions as a coin toss.

## Decision

### 1. A table, not a JSON document — and the rule that separates this from ADR-0042

The rule is not "files versus database". It is: **does the shape vary, and is it derived?**

| | Measurement table (ADR-0042) | Annotation (here) |
|---|---|---|
| Shape | varies by producer (ADR-0031: core plus blocks) | fixed: a box, a label, a source |
| Origin | derived — re-runnable | hand-made — irreplaceable |
| Written | once, whole, by one analysis | one at a time, edited, with undo behind it (M4-T08) |
| Therefore | a file | a row |

Rewriting a JSON document on every keystroke is the shape of a file lost to a crash mid-write. A
row per annotation is what a row is for, and it is what an undo stack can address.

ADR-0003's layout mentions "manual annotations (JSON)" — written before ADR-0031 existed and before
undo was scheduled. `annotations/` keeps its meaning for **painted masks**, which are bitmaps and
therefore files by that same ADR's own rule.

### 2. One shape: the box

Not a union of point, circle, polygon and mask.

A box is what a training set consumes — the only named consumer (M8) — and what a drag produces.
Every additional shape costs storage, an editor, a converter, and tests, for a reader nobody has
written. A circle converts to a box losslessly for training. If M6 finds an operator drawing
something a box cannot express, that shape then has a reader and this decision gets revisited.

**Masks stay deferred**, for the third time (ADR-0042 §3): painting is M6, and a format written
before its painter is written blind.

### 3. `source` says who drew it

`manual` or `from_detection`, with a `CHECK`.

Training a model on boxes copied from that model's own output is self-confirmation, and a training
set that cannot tell the two apart cannot avoid it. One column, two values, and the question it
answers is one M8 has to ask. The default is `manual`, which is the honest one: something a person
did.

### 4. Annotations cascade with their image — and the count must be askable first

`REFERENCES images(id) ON DELETE CASCADE`. A box pointing at an image the project no longer knows
about is not an annotation of anything.

But `remove_image` is an operator's explicit *"forget this scan"*, and it now discards hand work
that cannot be recreated. ADR-0040's argument for keeping a row does not apply — that was about a
row surviving a **missing file**, not about an operator deleting the row itself — so the answer is
not to refuse the deletion; it is that **`annotations_for` exists to be counted before the
deletion**, by a confirmation dialog that can say "this image has 12 annotations". Written here so
M6 inherits an obligation rather than discovering a data loss.

### 5. Coordinates are floats, and a zero-area box is refused

A drag is not on the pixel grid. Rounding is a decision the trainer makes with the whole box in
hand, not one the database makes on the way in.

`Detection.bbox` stays integer: a detector's output and a person's judgement are two different
things that happen to have four numbers each.

A box with `x2 <= x1` or `y2 <= y1` is refused by both the repository and a `CHECK`. It is a
mis-drag, and as a training example it is a picture of nothing.

### 6. No use case

`add_annotation` and its siblings are repository calls, and a function forwarding one call to one
object is ADR-0041's case for the fourth time.

The use case with policy in it — adopting a run's detections as a starting point for correction —
arrives with the editor that triggers it (M6), because *which* detections to adopt and what to
label them are questions about an interface that does not exist.

## Consequences

**Positive**

- The thing that cannot be recomputed is stored the way irreplaceable data should be: one row,
  addressable, editable in place, with an id that survives an edit.
- M8 can ask "which of these did a person actually draw?" — the question that keeps a training set
  honest.
- Undo (M4-T08) has something to address: `update_annotation` keeps the id, so undoing an edit is
  an edit back, not a resurrection.
- W9's blocker is half cleared: annotations exist and persist. Turning them into a dataset is M8.

**Negative**

- `remove_image` now destroys hand work, silently as far as this layer is concerned. Mitigated by
  §4's obligation on M6 rather than by a flag here — a `force=True` parameter would put the
  decision in the wrong layer.
- One shape means an operator who wants to circle something draws a box around it. Accepted, with
  the revisit condition named.
- The label is free text, so two spellings are two classes. A vocabulary belongs with the dataset
  that needs one (M8).

**Neutral**

- Schema version 3, the third step through ADR-0039's mechanism and the second applied to a
  database with rows in it.

## Alternatives considered

| Alternative | Why not |
|---|---|
| JSON files under `annotations/` | ADR-0003's original guess, written before the shape was fixed and before undo: a document rewritten per edit is a file lost mid-write, and it cannot be addressed by an undo stack |
| A polymorphic shape union | Four geometries, four editors, four converters, for a consumer that only reads boxes |
| Store annotations as `Detection` rows with a flag | Conflates what a model found with what a person decided — the exact distinction §3 exists to preserve |
| Refuse to remove an image that has annotations | Puts a UI decision in the storage layer, and blocks a legitimate action outright |
| A `force=True` on `remove_image` | A flag that exists so a caller can ignore a warning is a warning nobody reads |
| Integer coordinates | Rounds the operator's input on the way in, and the trainer is the one that knows what rounding it wants |

## Compliance

- `tests/integration/test_annotations.py` covers the round trip, the ordering, an edit that keeps
  its id, both refusals of a zero-area box, an unknown id on every method, survival across a
  session, and the cascade.
- One test drives a **v2** database up to v3 and asserts its rows survived.
- No annotation is stored with `x2 <= x1` or `y2 <= y1`: the repository refuses it and the schema
  refuses it again.
- `AnnotationSource` and the SQL `CHECK` carry the same two values; the enum is what the code uses.

## References

- ADR-0042 (the index is in the database, the measurement table is a file) — the decision §1
  distinguishes itself from
- ADR-0003 (projects are directories) — the layout whose "annotations (JSON)" this supersedes for
  boxes, and keeps for masks
- ADR-0041 (a use case earns its place) — §6
- ADR-0039 (the schema and its migrations) — the mechanism that carried v3
- `docs/Architecture.md` §2.3 (W9) · `docs/TASKS.md` M4-T07, M6, M8
