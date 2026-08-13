# CURRENT TASK

**ID:** `M7-T02`
**Title:** A box an operator drew, and the point they did not
**Milestone:** M7 — Annotation & metrology tools, second task
**Defect:** — · **ADR:** **ADR-0071**
**Branch:** `feat/m7-annotation-tools`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M7-T01 built the layer and put nothing in it. This is the task that lets an operator put something
there — the first time in this project's history that a person can *make* data rather than ask for
it to be computed.

`CommandStack` and `AddAnnotation` have been waiting since M4-T08 with tests as their only callers.

---

## The decisions this task has to make

**1. The point tool is not built, and the reason is the shape.**

ADR-0044 stores **one shape, the box**, and refuses a zero-area one twice — in the repository and in
a `CHECK`. A point has no extent, so a point tool must invent one: a "point size" control is that
invention wearing a label, and every row it writes claims an extent nobody measured.

ADR-0044 wrote the condition for revisiting this itself: *"if an operator draws something a box
cannot express, that shape then has a reader and this decision gets revisited."* A point has **no
reader** — M8's dataset builder consumes boxes. So the box tool ships and the point tool is closed
with this argument, the way M4 closed three of its own tasks.

**2. Drawing is a drag, and the drag stops panning while the tool is on.**

The view pans by dragging (M5-T05). A tool that draws *and* pans on the same gesture is a tool that
does the wrong one half the time, so turning the tool on turns panning off — visibly, with a checked
button.

**3. The label comes from a field, not a dialog.**

Annotating forty particles through forty modal dialogs is a feature nobody uses twice. The panel
carries the label to apply, and **an empty one is refused**: a box with no label is a rectangle
(ADR-0070), and the refusal happens here rather than as a database row saying `""`.

**4. Every box goes through the command stack.**

`CommandStack.do(AddAnnotation(...))` — because M4-T08 built undo for exactly this, and a drawing
tool without undo is a tool an operator cannot afford to be quick with. M7-T08 extends the wiring to
the rest of the tools; this task adds the menu and the first caller.

**5. A drag that is not a box is refused before the database sees it.**

The repository refuses a zero-area box, and so does a `CHECK` — but an operator who clicks by
accident should get nothing at all, not an error dialog. Below a few pixels the drag is discarded
silently; that is the same tolerance a click uses to be a click (ADR-0065).

---

## Scope

**In scope**

1. `gui/panels/viewer.py` — the drawing mode, the rubber band, and the box it emits
2. `gui/panels/annotate.py` — the label, the tool toggle, and what was drawn
3. `gui/viewmodels/session.py` — `add_annotation` through the command stack, `undo`, `redo`
4. `MainWindow` — an Edit menu with Undo/Redo, labelled by what they will do
5. **ADR-0071** — no point tool, drawing suspends panning, undo from the first tool
6. Tests: a drag becomes an annotation with the operator's label, undo removes it and redo restores
   **the same id**, an empty label and a tiny drag are refused, and the tool suspends panning

**Out of scope**

- **Polygon and brush** — M7-T03 and M7-T04, which need a shape the database does not have yet
- **Editing an existing annotation** — M7-T07
- **Wiring undo through the other tools** — M7-T08, once there are other tools

---

## Definition of done

- [x] A drag on the canvas becomes an annotation with the operator's label
- [x] Undo removes it; redo puts **the same row** back
- [x] An empty label and an accidental click are refused, quietly and loudly in the right order
- [x] ADR-0071 + the ADR index
- [x] `make check` green — 1179 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M7-T02: a box an operator drew, and the point they did not`

---

## What it turned up

**Nothing told the window that the *history* had moved.** The Undo item stayed dead after the first
box was drawn: the window refreshes its actions from the session's signals, and none of them meant
"a command was run". `annotations_changed` is wired to it now, which works **only because every
command in the stack mutates annotations today** — the first one that does not will need a signal
that says what actually changed. Written into the ADR as a neutral consequence rather than left as a
coincidence that happens to hold.
