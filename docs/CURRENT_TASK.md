# CURRENT TASK

**ID:** `M5-T05`
**Title:** A scan on screen, with the numbers that make it a measurement
**Milestone:** M5 — GUI shell, fifth task
**Defect:** — · **ADR:** **ADR-0056**
**Branch:** `feat/m5-gui-shell`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M5-T04 emits a selection and nothing listens. M5's second exit criterion is *"a scan renders with
correct nm axes and a scale bar"*, and this is the task that earns it — the first moment in the
project's history when an operator can **see their data** in this application.

---

## The decisions this task has to make

**1. What is shown: the file, or a processed version of it?** **The file.**

A tilted raw AFM map is harder to read than a flattened one, and every AFM tool flattens for
display. But flattening is an *analysis* — `flatten_plane` is science, it has an ADR, and its
output is what a run records. A viewer that silently flattens is a viewer showing something that is
not in the file, and an operator comparing it against a measurement would be comparing two
different arrays.

So: raw, labelled raw. A "flatten for display" toggle is a later task with a checkbox and a name.

**2. Where does the rendering happen?** In `application`, returning RGB.

The colormap lives in `infrastructure.imaging` (matplotlib), and `gui/` may not import
infrastructure. So a use case loads the image and another maps it to `uint8` RGB; the widget turns
an array into a `QImage` and draws it. The GUI decides *what* to show and *how big*; it never
decides how a value becomes a colour.

**3. Which widget?** `QGraphicsView`, not matplotlib.

Zoom and pan are what a graphics view is; matplotlib in a Qt canvas re-renders on every wheel event
and is the reason scientific GUIs feel slow. matplotlib stays where it belongs — in
`infrastructure/imaging/plots.py`, for figures that get saved.

**4. What makes it a measurement rather than a picture?** Three things, and they are the criterion:

- a **scale bar** in nanometres, sized to a round number and redrawn on zoom;
- a **coordinate readout** in nm *and* px, with the pixel's value;
- an honest answer when the scale is unknown — *"scale unknown"*, no bar, px only. ADR-0025 spent a
  milestone on absent-not-fabricated, and a viewer inventing 1 nm/px would undo it in one line.

**5. What does the LUT do?** Auto-ranges to the 2nd–98th percentile, with a control to take it to
the full range.

Auto because a single hot pixel flattens the whole image to grey, and percentiles are what every
SPM tool does; the full-range option because "what am I clipping?" is a question an operator must be
able to answer.

---

## Scope

**In scope**

1. `application/use_cases/display.py` — `load_for_display`, `render`, and the colormap list
2. `gui/panels/image_view.py` — `QGraphicsView` with wheel zoom, drag pan, readout signal
3. `gui/panels/viewer.py` — the panel: the view, a colormap box, the LUT control, the scale bar
4. `MainWindow` wiring: the selection from M5-T04 shows the image; the readout reaches the status bar
5. **ADR-0056** — raw not flattened, rendering in application, a graphics view, absent scale
6. Tests: rendering shape and dtype, percentile clipping, an unknown scale, the scale bar's round
   number, zoom limits, and the panel reacting to a selection

**Out of scope**

- **Detections and annotations drawn over the image** — they need M5-T06's viewmodel and M6's editor
- **A flatten toggle** — decision 1
- **Loading as a job** — M5-T07; a scan opens fast enough to be worth measuring before threading it

---

## Definition of done

- [x] Selecting an image in the explorer shows it
- [x] A scale bar in nm, a readout in nm and px, and "scale unknown" when it is
- [x] Rendering in `application`; no matplotlib and no infrastructure import in `gui/`
- [x] ADR-0056
- [x] Headless tests over the rendering and the widget
- [x] `make check` green — 926 tests, golden byte-identical
- [x] Docs, the ADR index, `Roadmap.md`
- [x] Commit: `M5-T05: a scan on screen, with the numbers that make it a measurement`

---

## What it turned up

**`load_afm`'s npy path leaked numpy's own `FileNotFoundError`.** PROJECT_RULES §3 forbids exactly
that — *"never let a NumPy/SciPy internal error escape as the public contract"* — and no caller
catching `NanoscopeError` would ever have seen it. Found by the viewer, whose entire error handling
*is* that distinction, and fixed at the **loader** rather than at the call site, so every caller
benefits rather than one.

**The scan rendered as a postage stamp.** `fitInView` at load time runs before the widget has its
final size, so it fits the image to a layout that has not happened yet. The view refits on resize
while it is still showing the whole scan, and leaves a zoomed view alone — a resize must not throw
away the operator's zoom.

**Seeing it was worth the two minutes.** Both findings above came from rendering a real phantom
into a real window and looking at the picture, not from the tests — which passed throughout.
