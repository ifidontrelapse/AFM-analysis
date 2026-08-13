# ADR-0063 — The overlay draws what the detection carries

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M6-T03)
- **Affects:** `gui/panels/viewer`, `gui/viewmodels` · M6

## Context

M6-T02 ends with *"30 detection(s)"* in a status bar and a scan on screen with nothing on it.
Looking at *where* a detector put its particles is the only way an operator can tell a good run from
a bad one — M3-T15 does the same job with numbers, and an operator cannot read a precision figure
off a scan.

`runs_for` has existed since M4-T05 with no reader at all: a scan analysed yesterday, selected
today, showed nothing.

## Decision

### 1. The shape is the one the detection carries

A **box** when `bbox` is present, a **circle** of `radius_px` when it is `None` (ADR-0031). Drawing
an invented box around a circle would be a shape nothing produced — the substitution ADR-0028
removed from `confidence`, one field over.

### 2. The overlay lives in the scene, in pixel coordinates

`QGraphicsView` transforms the scene, so an item at `(x_px, y_px)` stays on its particle at every
zoom and pan for free. The pen is **cosmetic**, so a 2 px circle is a one-pixel outline at 32× rather
than a filled blob. Painting over the viewport would mean redoing that arithmetic and being wrong at
the first drag.

### 3. The run shown is the newest stored one for the selected image

Loaded when the image is selected, replaced when a run is stored. M6-T09 owns *proving* it survives
a restart; being visible at all is this task's job.

### 4. One colour, and not the colormap's

A per-confidence ramp would be a second scale competing with the one that carries the measurement —
and the blob path has no confidence at all (ADR-0028), so half the overlay would be coloured by an
absence.

### 5. The count rides on the toggle

`Detections (30)`, on the checkbox that hides them. Six widgets share the viewer's control row and
the seventh was **clipped mid-word** in a real window; "Detections (0)" and an unticked box also say
two different things, which a separate label had to spell out.

## Consequences

**Positive** — a run is visible where it happened; the toggle answers *"what does this look like
without the circles?"*, which is the question behind every argument about a false positive; stored
runs stop being invisible.

**Negative** — the overlay is drawn item by item, which is fine at hundreds of particles and will not
be at a hundred thousand; the trigger for a single custom item that paints them all is a scan that
stutters. Nothing is selectable yet either: M6-T05 needs that for the table and adds it there.

**Neutral** — **the circle branch has no producer in this application today.** The blob detector
synthesises a bbox (found in M3-T24, recorded again in M4-T05), so every detection currently draws
as a box. The branch stays, because the entity says the field is optional and the next detector may
mean it.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Draw a box for everything | An invented box is a shape nothing found |
| Paint the overlay in `paintEvent` | Re-implements the view's transform, wrongly, at the first pan |
| Colour by confidence | A second scale beside the data's, and absent on half the detections |
| Show every run at once | Two runs of the same scan differ *by* their detections; drawing both is asking which |

## Compliance

`tests/gui/test_detection_overlay.py` pins the shape choice, the cosmetic pen, scene coordinates,
the count on the toggle, that selecting another image clears the overlay, that the newest stored run
is shown on selection, and that a new run replaces the old one.

## References

- ADR-0031 (`bbox` is absent, not empty) and ADR-0028 (no invented confidence) — §1
- ADR-0042 — the stored run this reads
- ADR-0056 / ADR-0061 — the viewer's honesty rules the overlay inherits
