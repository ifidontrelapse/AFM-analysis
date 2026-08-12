# ADR-0056 — The viewer shows the file, and says what it cannot know

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M5-T05)
- **Affects:** `application/use_cases/display`, `gui/panels` · M5 · M6's annotation editor

## Context

M5's second exit criterion: *"a scan renders with correct nm axes and a scale bar"*. M5-T04 emits a
selection and nothing listens. This is the first moment in the project's history when an operator
can **see their data** in this application.

## Decision

### 1. The viewer shows what is in the file — raw, not flattened

Every SPM tool flattens for display, and a tilted AFM map is genuinely harder to read. But
flattening is an **analysis**: `flatten_plane` is science, it has ADR-0029 behind it, and its output
is what an analysis run records.

A viewer that silently flattens shows something the file does not contain, and an operator
comparing what they see against a measured height would be comparing two different arrays. Raw, and
labelled as such; a "flatten for display" toggle is a later task, with a checkbox and a name.

### 2. Rendering happens in `application`, not in the widget

`load_for_display` reads the file; `render` maps values to RGB. The colormap lives in
`infrastructure.imaging` (matplotlib), which `gui/` may not import — and **how a value becomes a
colour is not a widget's decision** anyway. The widget decides what to show and how big.

### 3. `QGraphicsView`, not matplotlib

Zoom and pan are what a graphics view *is*. A matplotlib canvas re-renders the whole figure on
every wheel event, which is why scientific GUIs feel slow. matplotlib stays in
`infrastructure/imaging/plots.py`, for figures that get saved.

### 4. Three numbers make it a measurement rather than a picture

- a **scale bar** at a round length — a bar reading "137 nm" is one nobody can measure against by
  eye, which is the only thing a scale bar is for;
- a **readout in nm *and* px**, with the value under the cursor: a pixel index is what you click, a
  nanometre is what you report;
- **an honest absence.** No scale means no bar, no nm in the readout, and the words "scale unknown".
  ADR-0025 spent a milestone on absent-not-fabricated, and a viewer writing "1 nm/px" would undo it
  in one line. This is the last surface that could have broken that rule.

### 5. The value window is percentile by default

The 2nd–98th percentile, with a control for the full range. One hot pixel otherwise flattens the
whole image to grey — the phantom in the tests has one for exactly that reason — and *"what am I
clipping?"* has to remain answerable.

### 6. A refusal is a readout line, not a dialog

A missing file, or a format with no reader, becomes a message in the status bar. The operator
clicked a row in a list, not a button labelled "load"; a modal box for a side effect of selection
is a modal box in the way.

## Consequences

**Positive**

- M5's second exit criterion is met, and verified by rendering a characterization phantom into a
  real window: 24 particles, a 100 nm bar, `x=100 y=50 px (200.0, 100.0) nm value=0.01413`.
- The GUI still imports no `infrastructure`, so the guard from M4-T15 stays green.
- An unknown scale is visible as an unknown scale, everywhere it appears.

**Negative**

- A tilted scan looks tilted, and an operator used to Gwyddion will want the toggle. §1 is that
  trade, made deliberately and reversible in one checkbox.
- Rendering is synchronous. A 4096² scan takes a noticeable moment; M5-T07 has the job runner, and
  moving this onto it is a change to one call.

**Neutral**

- Six colormaps rather than matplotlib's hundreds. A list nobody reads is a list nobody uses.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Flatten for display | Shows something the file does not contain, and disagrees with the measurement |
| matplotlib canvas | Re-renders the figure on every wheel event |
| Render in the widget | `gui/` would import `infrastructure`, and colour mapping is not a widget's decision |
| A bar at an arbitrary length | Nobody can measure "137 nm" against a bar by eye |
| Assume 1 nm/px when unknown | The exact substitution ADR-0025 spent a milestone deleting |
| Full range by default | One hot pixel turns the scan grey |
| A dialog when an image cannot be shown | A modal box triggered by clicking a list row |

## Compliance

- `tests/gui/test_viewer.py` covers loading (raw, the project's scale, an unknown scale), rendering
  (shape, dtype, the percentile window, the full range, an unknown colormap refused, every offered
  colormap), the scale bar (round, scaling, absent without a scale), the panel (size announced, nm
  and px readout, no nm without a scale, a refusal as a message) and the zoom limits.
- Verified in a real window under `QT_QPA_PLATFORM=offscreen`, with a phantom from the
  characterization suite.

## References

- ADR-0025 (an unknown scale is not a fabricated one) — §4's absence
- ADR-0029 (`flatten_lines` promotes like `flatten_plane`) — the science §1 declines to do silently
- ADR-0053 / ADR-0055 — the window and the panel this sits beside
- `docs/Roadmap.md` M5 exit criteria
