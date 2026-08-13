# ADR-0070 — Hand work is drawn as hand work

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M7-T01)
- **Affects:** `gui/panels/viewer`, `gui/viewmodels` · M7 · M8

## Context

M4-T07 made an annotation a row **because it cannot be recomputed** (ADR-0044). M4-T08 built undo
around it. M5-T04's confirmation counts them before an image is removed. And nothing in a window had
ever drawn one: the most expensive data in a project was the only data with no representation on
screen.

Every tool in M7 puts annotations *into* this layer, so the layer comes first.

## Decision

### 1. An annotation is not a detection, and does not look like one

A detection is what a machine found; an annotation is what a person judged. Different colour, its own
toggle, its own count, and drawn **above** the detections — because it is what the operator is
working on, and what a click should reach first when M7-T02's tools arrive.

### 2. The two sources are visibly different

`source` is `manual` or `from_detection`, and ADR-0044 made that distinction load-bearing for
training: *a model trained on its own output is confirming itself*. A screen that draws them alike
undoes that in the one place an operator would have noticed. Manual is solid; adopted is dashed, in
the colour this palette uses for *look at this*.

A test asserts **every** `AnnotationSource` has a style, because a source with no entry raises while
drawing — a crash in the layer whose whole job is to be trusted about provenance.

### 3. The label is the operator's own text, and does not scale

A box with no label is a rectangle; the label is why the box exists. It is drawn with
`ItemIgnoresTransformations`, because a label that grows to fill the screen at 32× is a label nobody
can read at 32×.

### 4. The session loads them, on selection, like the run

`annotations_for` has had one caller since M4-T07 — a dialog that counts them without showing one.

### 5. This task mutates nothing

No command reaches the stack. Drawing is M7-T02, editing is M7-T07, and a layer that could already
change what it displays would make both of those harder to test.

## Consequences

**Positive** — hand work is visible where it was done; the provenance distinction M8 depends on is
visible too; the tools in the rest of M7 have somewhere to put what they create.

**Negative** — annotations are drawn item by item, like the detections, which is fine at hundreds and
will not be at a hundred thousand. Labels overlap when boxes do; the fix is a layout nobody has asked
for, and the toggle answers the immediate version of the problem.

**Neutral** — no selection or hit-testing yet. The overlay is above the detections precisely so that
M7-T02 can add it without reordering anything.

## Alternatives considered

| Alternative | Why not |
|---|---|
| One colour for detections and annotations | Erases the machine/person distinction M8 depends on |
| Hide the source difference behind a tooltip | A tooltip is a thing you find; provenance has to be a thing you see |
| Labels that scale with the scene | Unreadable at both ends of a 2000× zoom range |
| Load annotations lazily, on first draw | The panel would decide when to read the database |

## Compliance

`tests/gui/test_annotation_layer.py` asserts they load on selection and belong to their own image,
that the box lands where it was drawn, that the two sources differ in **both** colour and line style,
that every source has a style, that annotations sit above the detections, that the label is the
operator's text and ignores the view transform, and that the toggle empties the layer while keeping
the count.

## References

- ADR-0044 — an annotation is a row, and why `source` matters
- ADR-0063 — the detection overlay this sits above
- ADR-0057 — the session that holds what the panels draw
