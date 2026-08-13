# CURRENT TASK

**ID:** `M6-T03`
**Title:** The detections, drawn where they were found
**Milestone:** M6 — Analysis workflow in the GUI, third task
**Defect:** — · **ADR:** **ADR-0063**
**Branch:** `feat/m6-analysis-workflow`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M6-T02 ends with *"30 detection(s)"* in a status bar and a scan on screen with nothing on it. Looking
at where a detector put its particles is the only way an operator can tell a good run from a bad one
— it is what the whole milestone is for, and it is what the evaluation harness (M3-T15) does with
numbers instead of eyes.

---

## The decisions this task has to make

**1. What is drawn?** What each detection actually carries, and nothing inferred.

A `bbox` is `None` on the LoG path (ADR-0031) and present for a box detector, so: **a box when there
is one, a circle of `radius_px` when there is not.** Drawing an invented box around a circle would be
a shape the detector never produced — the same class of substitution ADR-0028 removed from
`confidence`.

**2. Where does the overlay live?** In the scene, in pixel coordinates.

`QGraphicsView` transforms the scene, so an item placed at `(x_px, y_px)` stays on its particle at
every zoom for free, and the pen is cosmetic so a circle does not turn into a blob at 32×. A
`paintEvent` drawing over the viewport would have to redo that arithmetic, and would be wrong at the
first pan.

**3. Which run is shown?** The newest one for the selected image, loaded when it is selected.

`runs_for` has existed since M4-T05 and nothing has read it. Selecting a scan that was analysed
yesterday and seeing nothing would make the stored run invisible — M6-T09 owns *proving* that across
a restart; showing it at all is this task's job.

**4. Can it be turned off?** Yes, and that is not decoration: the overlay covers the data it
describes, and *"what does this look like without the circles?"* is a question an operator asks
about every false positive.

**5. Does the overlay colour mean anything?** It is one colour, and it is not the colormap's.

A per-confidence colour ramp would be a second scale on screen competing with the one that carries
the measurement, and the LoG path has no confidence at all (ADR-0028) — so half the detections would
be coloured by an absence.

---

## Scope

**In scope**

1. `gui/viewmodels/session.py` — the current run, loaded on selection and replaced when one is
   stored
2. `gui/panels/viewer.py` — the overlay items, the toggle, and the count
3. **ADR-0063** — what is drawn, where it lives, and which run it belongs to
4. Tests: a box detection draws a box, a circle detection draws a circle, the overlay follows the
   selection, the newest stored run is shown, the toggle empties the scene, and a new run replaces
   the old one

**Out of scope**

- **Selecting a detection** — M6-T05 needs it for the table, and builds it there
- **Masks** — M6-T04
- **Editing a detection** — M7's annotation tools

---

## Definition of done

- [x] Detections from the newest run are drawn over the scan, at every zoom
- [x] A box is drawn when the detector produced one, a circle when it did not
- [x] The overlay can be turned off, and says how many it is showing
- [x] ADR-0063 + the ADR index
- [x] `make check` green — 1084 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M6-T03: the detections, drawn where they were found`

---

## What it turned up

**The circle branch has no producer in this application.** The blob detector synthesises a bbox —
recorded in M3-T24 and again in M4-T05 — so every detection currently draws as a box. The branch
stays: the entity says the field is optional, and the next detector may mean it.

**The count label was the seventh widget in the viewer's control row, and clipped mid-word.** Seen
in the window. It rides on the toggle now — `Detections (30)` — which also says something a separate
label had to spell out: an unticked box is "hidden", `(0)` is "found none".
