# ADR-0071 — The shape is a box, and a tool does not invent one

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M7-T02)
- **Affects:** `gui/panels/annotate`, `gui/panels/viewer`, `gui/viewmodels` · M7 · M8

## Context

M7-T01 built the annotation layer and put nothing in it. This is the first surface in the project
where an operator **makes** data rather than asking for it to be computed — and `CommandStack` and
`AddAnnotation` have been waiting since M4-T08 with tests as their only callers.

The task was scheduled as *"point and box tools"*.

## Decision

### 1. The point tool is not built

ADR-0044 stores **one shape, the box**, and refuses a zero-area one twice — in the repository and in
a `CHECK`. A point has no extent, so a point tool must invent one, and a *"point size"* control is
that invention wearing a label: every row it writes would claim an extent nobody measured.

ADR-0044 wrote the condition for revisiting this itself — *"if an operator draws something a box
cannot express, that shape then has a reader and this decision gets revisited"* — and a point has no
reader: M8's dataset builder consumes boxes. So the box tool ships and the point tool is closed with
this argument, the way M4 closed three of its own tasks and M6-T04 closed its panel.

### 2. Drawing suspends panning, visibly

The view pans by dragging (M5-T05). A tool that draws *and* pans on the same gesture does the wrong
one half the time, so turning the tool on switches the drag mode and the cursor. The button stays
checked while it is on.

### 3. The label comes from a field, and an empty one is refused

Annotating forty particles through forty modal dialogs is a feature nobody uses twice. The panel
carries the label to apply; an empty one is refused **with a sentence**, because a box with no label
is a rectangle (ADR-0070) and the refusal belongs here rather than as a row saying `""`.

### 4. A drag too small to be a box is discarded silently

The repository refuses a zero-area box and so does the `CHECK` — but an operator who clicked by
accident should get nothing at all, not an error dialog. Below three pixels the drag is dropped, the
same tolerance a click uses to be a click (ADR-0065).

**Loudly and quietly, in that order:** a wrong label is a mistake worth telling somebody about, and a
slipped click is not.

### 5. Every box goes through the command stack

`CommandStack.run(AddAnnotation(...))`. M4-T08 built undo for exactly this, and its promise — *redo
puts **the same row** back, because a fresh id leaves every command above it pointing at nothing* —
now has a caller outside its own tests. The Edit menu is labelled by what Undo would take back:
*"Undo alone makes an operator press it to find out."*

## Consequences

**Positive** — an operator can correct the machine; undo covers the correction from the first tool
rather than being retrofitted in M7-T08; the annotation layer M7-T01 built has a producer.

**Negative** — one label applies to every box until the field is changed, which is right for a batch
of one kind of particle and wrong for a scan with three. Changing a label after the fact is M7-T07's
editing, and until then the answer is to draw one kind at a time.

**Neutral** — the window refreshes the Edit menu on `annotations_changed`, because every command in
the stack mutates annotations *today*. The first command that does not needs a signal that says "the
history moved" (M7-T08).

## Alternatives considered

| Alternative | Why not |
|---|---|
| A point tool storing a small box | Claims an extent nobody measured; the invention with a label |
| A point tool storing a zero-area box | Refused twice, by the repository and a `CHECK` — correctly |
| A dialog per box | Forty particles, forty dialogs |
| Draw without suspending panning | The same gesture doing the wrong thing half the time |
| Refuse a tiny drag with a message | A slipped click is not a mistake worth a dialog |
| Add annotations directly, undo in M7-T08 | A drawing tool nobody can afford to be quick with |

## Compliance

`tests/gui/test_box_tool.py` asserts a drag becomes an annotation with the field's label and is
normalised however it was dragged, that **undo removes it and redo restores the same id**, that the
menu names what it would take back, that an empty label is refused with a sentence and a tiny drag
silently, that the tool suspends panning, and that the panel offers exactly one tool — with a
separate test showing the repository refusing the zero-extent box a point would be.

## References

- ADR-0044 — one shape, and the condition for revisiting it
- ADR-0045 / M4-T08 — undo, and why redo restores the same row
- ADR-0070 — the layer this fills, and the label that makes a box an annotation
- ADR-0065 — the click tolerance §4 reuses
