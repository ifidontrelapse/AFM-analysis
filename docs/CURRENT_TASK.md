# CURRENT TASK

**ID:** `M7-T07`
**Title:** Correcting the machine without rewriting what it did
**Milestone:** M7 — Annotation & metrology tools, seventh task
**Defect:** — · **ADR:** **ADR-0076**
**Branch:** `feat/m7-annotation-tools`
**Status:** **done 2026-08-14.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

The task list calls this *"manual add/edit/delete of detections"*, and reading it properly is most of
the work — because **a detection is not editable, and ADR-0044 already said what to do instead.**

Three things M4 built are still waiting for a caller: `AnnotationSource.FROM_DETECTION` has never
been written by anything, and `UpdateAnnotation` and `RemoveAnnotation` have no callers outside their
own tests.

---

## The decisions this task has to make

**1. A detection is not edited. It is adopted.**

A stored detection is *what a detector produced in a run* — a record of something that happened
(ADR-0042). An operator deleting one makes the run describe an analysis that never ran, and the
project loses the only honest answer to *"what did the detector actually find?"*, which is the
question M3-T15's evaluation harness exists to ask.

ADR-0044 built the way out and named it: an annotation carries `source`, `manual` or
`from_detection`, *because* **"a model trained on its own output is confirming itself"**. Correcting
the machine means **adopting** a detection into an annotation — and the adopted one is marked, for
ever, as having come from the machine.

**2. What is editable is an annotation**, which is the one thing in a project that was always
hand-made (ADR-0044). Its label, its box, and its existence.

**3. Adoption is one click, and adopting everything is one more.**

Reviewing forty detections means keeping thirty-eight and fixing two; a workflow that costs a
dialog per particle is one nobody uses. Both go through the command stack, so both undo.

**4. Deleting an annotation asks nothing.**

M5-T04's confirmation exists because removing an *image* destroys hand work that cannot be recomputed
(ADR-0044). Deleting **one box** an operator is looking at, with `Ctrl+Z` in the same menu, is the
opposite case: a dialog there is the one nobody reads by the third time.

**5. Editing needs a selection, and the canvas already has one.**

M6-T05 put particle selection in the viewmodel; annotations get the same treatment, and a click
reaches the annotation layer first because M7-T01 drew it on top for exactly this.

---

## Scope

**In scope**

1. `gui/viewmodels/session.py` — `adopt_detection`, `adopt_all_detections`, `rename_annotation`,
   `remove_annotation`, and the selected annotation
2. `gui/panels/viewer.py` — clicking an annotation selects it; the selected one is thicker
3. `gui/panels/annotate.py` — adopt, rename, delete
4. **ADR-0076** — a detection is a record, an annotation is the edit, and adoption is marked
5. Tests: an adopted detection is `from_detection` and matches its box, adopt-all, rename and delete
   through the stack with undo, and the selection

**Out of scope**

- **Editing a polygon's vertices or a painted mask** — the shapes are stored, and a vertex editor is
  a tool of its own that nothing has asked for
- **Deleting a stored run** — a destructive repository operation with a confirmation behind it
- **Undo across the *other* tools** — M7-T08, which is now mostly an audit

---

## Definition of done

- [x] A detection can be adopted as an annotation, marked `from_detection`
- [x] An annotation can be renamed and deleted, both undoable
- [x] No stored detection is ever modified
- [x] ADR-0076 + the ADR index
- [x] `make check` green — 1264 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M7-T07: correcting the machine without rewriting what it did`

---

## What it turned up

**The task's own title was the decision.** *"Manual add/edit/delete of detections"* describes an
operation this project deliberately cannot perform, and ADR-0044 had already written what to do
instead — four milestones earlier, in the ADR that created `source`. Three pieces of M4 were waiting
for exactly this task and none of them had ever been called: the `FROM_DETECTION` mark, and the
update and remove commands.

**A blob detection has no box to adopt.** `bbox` is `None` on that path (ADR-0031), so the adopted
annotation takes the square bounding the circle — which is not an invention this task made up but
ADR-0044's own sentence: *"a circle converts to a box losslessly for training."* The rule was written
down before the code that needed it.
