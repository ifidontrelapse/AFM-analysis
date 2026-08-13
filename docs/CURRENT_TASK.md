# CURRENT TASK

**ID:** `M7-T01`
**Title:** The hand work, on screen and told apart from the machine's
**Milestone:** M7 — Annotation & metrology tools, first task
**Defect:** — · **ADR:** **ADR-0070** (to be written)
**Branch:** `feat/m7-annotation-tools`
**Status:** planning 2026-08-13.

---

## Why this task is first

M4-T07 made an annotation a row **because it cannot be recomputed** (ADR-0044). M4-T08 built undo
around it. M5-T04's confirmation counts them before an image is removed. And **nothing in a window
has ever drawn one** — the most expensive data in a project is the only data with no representation
on screen.

Every tool in M7 puts annotations *into* that layer, so the layer comes first.

---

## The decisions this task has to make

**1. Are annotations detections?** No, and they must not look like them.

A detection is what a machine found; an annotation is what a person judged. ADR-0044 made the
distinction load-bearing for training — *"a model trained on its own output is confirming itself"* —
and a screen that draws them in one colour undoes that in the only place an operator would notice
it. Different colour, drawn **above** the detections, and its own toggle.

**2. What does an annotation adopted from a detection look like?**

`source` is `manual` or `from_detection` (ADR-0044 §3), and that distinction exists *because* the
two must not be confused. So it is visible: a box an operator drew and a box they accepted from the
machine are not the same claim, and M8 will care about which is which.

**3. Where does the layer live?** In the same scene as everything else.

The overlay stack, bottom to top: the scan, the masks, the detections, the annotations. Hand work on
top, because it is what the operator is working on and what a click should reach first when M7-T02
arrives.

**4. Who loads them?** The session, on selection, like the run.

`annotations_for` has had one caller since M4-T07 — M5-T04's confirmation dialog, which counts them
without ever showing one.

**5. Does the label show?** Yes, above the box, and it is the operator's own text.

An annotation's label is why it exists; a box with no label is a rectangle. It is drawn small and in
the annotation's colour, and it does not scale with the zoom — a label that grows to fill the screen
at 32× is a label nobody can read at 32×.

---

## Scope

**In scope**

1. `gui/viewmodels/session.py` — the selected image's annotations, and a signal
2. `gui/panels/viewer.py` — the annotation layer, its labels, its toggle, and the stacking order
3. **ADR-0070** — annotations are not detections, and the screen says which is which
4. Tests: they are loaded on selection, drawn above the detections in their own colour, the two
   sources are distinguishable, the label is the operator's text, and the toggle empties the layer

**Out of scope**

- **Drawing a new one** — M7-T02's tools, which this layer exists to receive
- **Editing or deleting from the canvas** — M7-T07
- **Anything that mutates** — this task adds no command to the stack

---

## Definition of done

- [ ] Annotations of the selected image are drawn, labelled, above the detections
- [ ] A manual annotation and one adopted from a detection are distinguishable
- [ ] The layer has its own toggle and its own count
- [ ] ADR-0070 + the ADR index
- [ ] `make check` green, golden byte-identical
- [ ] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [ ] Commit: `M7-T01: the hand work, on screen and told apart from the machine's`
