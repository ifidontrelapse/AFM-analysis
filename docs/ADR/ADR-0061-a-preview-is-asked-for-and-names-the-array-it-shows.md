# ADR-0061 — A preview is asked for, and names the array it shows

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M6-T01)
- **Affects:** `application/use_cases`, `gui/panels`, `gui/viewmodels` · M6

## Context

M6's exit criterion is *load → detect → segment → measure → export, entirely through the UI*, and
preprocessing is the step every later one stands on: `run_analysis` levels a scan and builds its
substrate before it detects anything, and that substrate decides what counts as a particle.

Two things were waiting for this task.

**ADR-0056 named it.** The viewer shows the file, raw and unflattened, *"because flattening is an
analysis"* — and the ADR ended: *"a 'flatten for display' toggle is a later task with a checkbox and
a name."*

**`run_preprocessing` took no parameters.** `build_substrate_map` has three that matter — a physical
minimum size (ADR-0024), a manual opening radius that *is* the radius when given (ADR-0014), and an
opening scale measured in ADR-0037 — and none of them could be reached from anywhere above.

The roadmap also sets the rule for the whole milestone: **the UI must not introduce its own
defaults.**

## Decision

### 1. The parameters are pass-through, and the defaults are named once

`run_preprocessing` and the new `preprocess_image` take `min_size_nm`, `manual_radius_px` and
`opening_scale`, with the values the function already used. `DEFAULT_MIN_SIZE_NM` and
`DEFAULT_OPENING_SCALE` are named in `application/use_cases/preprocessing.py` so a panel shows the
value it will actually get instead of typing the number into a spin box — a literal in a widget is
precisely the second place a default starts living.

`DEFAULT_MIN_SIZE_NM` mirrors a bare `5` in a science signature this task does not rewrite
(PROJECT_RULES §4.1), and **a test compares the two through `inspect.signature`**, so the mirror
cannot drift silently.

The compliance test for the whole rule is dull on purpose: an untouched panel and a bare
`run_preprocessing` call are compared **array for array**.

### 2. `preprocess_image(repository, image_id, …)` resolves the row, not the panel

The same resolution `run_analysis` does: the record, the path *through the repository*, and **the
scale the project recorded**. A panel assembling a path, a format and a scale of its own is exactly
where M4-T05 found the D-07 family of defect reintroduced one layer up.

The format helper moved from `analysis.py` to `preprocessing.py` — the module that loads AFM files —
rather than being copied into a second caller.

### 3. The preview is asked for, and runs as a job

A button, not a live re-run. Preprocessing a 4096² scan is seconds of NumPy (Architecture §4.5), and
a pipeline that re-runs on every spin-box keystroke is a UI that fights the operator and heats their
laptop. It goes through M5-T07's runner, which is its second consumer and the reason it exists.

### 4. The viewer draws any stage, and says which one

Raw, flattened, substrate, result — and the stage is named on screen, next to the colormap, with the
long description as its tooltip.

**This does not weaken ADR-0056; it is what ADR-0056 actually said.** The rule was never "show the
file and nothing else", it was *never show something the file does not contain without saying so*.
A label is what makes the difference between a viewer that shows a derived array and a viewer that
pretends a derived array is the scan.

The plane-only intermediate is not offered: `PreprocessingResult` does not keep it, and adding a
field to an entity for a preview would change what a *run* records.

### 5. A preview is not a result, and is not stored

`run_analysis` records a run with its detections and its measurement table (ADR-0042). A preview is
a look at intermediate arrays. Writing rows for it would make *"what runs does this image have?"* a
question about which buttons somebody pressed, and M6-T09 owns persistence.

It also **belongs to the scan it was computed from**: selecting another image drops it, because a
substrate map from one scan drawn over another is the worst version of this feature.

### 6. The panel reports what was used, not what was asked for

The opening radius the run actually used, the Otsu typical radius in px *and* nm when the scale is
known, and the number of objects the estimate **kept**. ADR-0014 and ADR-0017 both end on that
distinction, and all three numbers are already in the result — nothing here recomputes them.

## Consequences

**Positive**

- The first analysis step is reachable from a window, with the trade-off ADR-0037 measured exposed
  rather than hidden inside a branch.
- ADR-0056's deferred toggle arrives as something better: not "flatten for display" but *which stage
  am I looking at*.
- `run_analysis` and a panel now resolve an image the same way, through one function.
- The golden is untouched, and a test proves the panel's blank state reproduces it array for array.

**Negative**

- A preview costs a full preprocessing pass per press, with no caching. Two presses with the same
  numbers do the work twice. Caching needs a key over the parameters *and* the file, which is a
  decision M6-T09's persistence should make rather than this task inventing one.
- The parameters are not remembered — not in the project, not per image. They are the analysis
  parameters M6 will store with a run, and storing them twice, in two shapes, would be worse than
  storing them once.
- The stage selector lives in the viewmodel and is driven from panels; there is no combo box for it
  yet. Choosing a stage is currently a call, not a control — M6-T03's overlay work is where the
  viewer's own controls get revisited.

**Neutral**

- The "Flatten and level (always)" checkbox is checked and disabled. There is no version of this
  step that skips levelling — a substrate is estimated from a levelled map — and showing it keeps
  the pipeline readable in order rather than implying a choice that does not exist.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A live preview on every parameter change | Seconds of NumPy per keystroke; a UI that fights the operator |
| Run preprocessing on the main thread | The freeze the job abstraction exists to prevent |
| Defaults typed into the spin boxes | A second place where a default lives; M6's rule forbids it |
| A "flatten for display" checkbox, as ADR-0056 imagined | Names one stage of four, and hides that the substrate is a stage too |
| Store the preview as a run | A run is what `run_analysis` records; buttons pressed are not results |
| Keep the preview across a selection change | A substrate map from another scan, drawn over this one |
| Add the plane-only array to `PreprocessingResult` | Changing what a run records, for a preview |

## Compliance

- `tests/gui/test_preprocessing_panel.py::TestTheDefaultsAreNotTheUIs` compares an untouched panel
  against a bare `run_preprocessing` call array for array, and checks the named default still equals
  the science's own.
- The same file pins each parameter reaching `build_substrate_map` (a manual radius **is** the
  radius; the scale moves it), that nothing runs until the button is pressed, that a failure is a
  message and not a preview, that the viewer names the stage, that a stage with no preview behind it
  is refused, and that nothing is written into `results/`.
- The golden is byte-identical: this task added parameters and moved no number.

## References

- ADR-0056 §1 — the deferred toggle this replaces, and the rule §4 keeps
- ADR-0014, ADR-0017, ADR-0024, ADR-0037 — the four decisions the three parameters carry
- ADR-0042 — what a *run* is, and why a preview is not one
- ADR-0043 / ADR-0058 — the job and the marshalling §3 uses
- `docs/Roadmap.md` M6 — *the UI must not introduce its own defaults*
