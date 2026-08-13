# ADR-0057 — One viewmodel, and a widget that emits intent

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M5-T06)
- **Affects:** `gui/viewmodels`, `gui/panels`, `gui/main_window` · M5 · M5-T07's job runner

## Context

ADR-0055 §4 declined a viewmodel and attached a condition to the refusal:

> *M5-T06 owns the viewmodel layer. Inventing half of one for a list and two actions is the
> abstraction this project has declined at every previous opportunity … **When M5-T05's viewer needs
> the same selection, there will be two consumers and a reason.***

M5-T05 shipped that viewer. Three facts about the code as it stood:

1. **Both panels held the composition root.** `ProjectExplorer(app)` read `app.repository`, called
   the `open_project` use case and `remove_image`; `ImageViewer(app)` called `load_for_display`.
   *Which* image to read when a row is clicked is a decision, and Architecture §2.3 says a widget
   does not make them.
2. **A widget was wired to a widget** — `explorer.image_selected.connect(viewer.show_image)`. With
   two panels that is one line. With the Properties panel it is three, and every panel after that
   pays the same tax: n panels, n² connections, and no single object that can answer "what is the
   session showing right now?".
3. **The Properties dock had nothing to fill itself from.** It describes the array the viewer has
   *already loaded*; without somewhere to hold it, the honest implementations are "read the file a
   second time" or "ask the viewer widget", and the second is the wiring problem again.

M5-T07 adds a fourth. ADR-0043 states three times that a job's listener fires **on the worker
thread**, and a widget method called from a worker thread is a crash that happens later, somewhere
else.

## Decision

### 1. One viewmodel, not one per view

`SessionViewModel` holds what more than one panel needs: the open project, the selected image id,
and the loaded `DisplayImage`. State only one widget can want — the viewer's colormap, its
full-range checkbox, its zoom — stays in that widget.

This is ADR-0041's rule at a different altitude: *a layer earns its place or is not written.* What
earns it here is shared state and cross-thread signalling; a viewmodel per view would be four
classes forwarding to one, written for symmetry with a diagram.

The trigger for a second one is stated so its absence stays a decision: **a view whose own state
another view needs to read.**

### 2. Intent goes in as a call, state comes out as a signal

A panel calls `session.select_image(7)`. The session loads, and emits `image_changed`. Every panel
that shows an image hears it — including panels that do not exist yet.

**No panel connects to another panel.** Each subscribes to the session and to nothing else, which
is n connections rather than n², and it means a new panel is added without editing the old ones.

### 3. The image is loaded once, in the viewmodel

The viewer draws the array and the properties panel describes it. Loading it in each would cost a
disk read per selection, and — the reason that matters — would make it possible for two panels to
disagree about the same scan.

### 4. The viewmodel holds no widget and opens no dialog

It is testable with no `QWidget` constructed at all, and `tests/gui/test_session_viewmodel.py`
builds none.

The confirmation ADR-0055 designed stays in the panel that asks it: *whether* to ask, and in what
words, is a view decision. The viewmodel supplies the count (`annotation_count`) and performs the
removal (`remove_image`).

### 5. A refusal is one `failed(str)` signal, and the dialog stays where the button was

The window puts every refusal in the status bar. A modal dialog appears only for an action the
operator asked for by name — opening a project. A failure to *display* a scan is a status line,
because they clicked a row, not a button labelled "load" (ADR-0056, now stated once rather than per
panel).

### 6. The selection survives a failed load

`image_id` is set even when the file cannot be read; `image` is `None`. A scan whose file is missing
is still the row the operator clicked, and forgetting it is the likeliest thing they want to do
next — so *Remove* follows the selection, not the load.

### 7. It is a `QObject`, and that is the point for M5-T07

A queued signal is how Qt moves a call from a worker thread to the thread the widgets live on.
M5-T07 marshals ADR-0043's progress callbacks onto this object. Nothing here does that yet; what
this decision does is make the object it will marshal onto exist, rather than leaving M5-T07 to
invent one under time pressure.

## Consequences

**Positive**

- No module under `gui/panels/` imports `nanoscope.app` — checked by a test, not by review.
- Adding a panel is one subscription, not one connection per existing panel.
- One file read per selection, for any number of panels showing it.
- The session's behaviour is tested headless and widget-free, which is the half of the GUI worth
  regression-testing.
- M5-T07 has a main-thread `QObject` waiting for it.

**Negative**

- One more indirection between a click and a use case. On a two-panel application that is a cost
  paid before the benefit is fully visible; the Properties panel is the first instalment of the
  benefit and the job runner is the rest.
- The window now holds `_last_failure` so it can put the most recent refusal into a dialog. A
  signal that carries a message and a caller that needs it *after* the call is a small seam; the
  alternative was raising across the layer, which §5 exists to avoid.

**Neutral**

- The explorer lost its `image_selected` signal. A selection is an intent, and the panels that care
  hear about the *result* from the session — so the signal had no subscriber left.

## Alternatives considered

| Alternative | Why not |
|---|---|
| One viewmodel per view | Four classes forwarding to one; symmetry with a diagram is not a requirement |
| Keep panels on the container | Every panel decides what to load, and n² wiring grows with every panel |
| A signal bus / mediator with no state | The state is the point: "which image is selected" has to live somewhere |
| Let the viewmodel own the confirmation dialog | A viewmodel that needs a window to be tested has a window inside it |
| Let each panel load the image it needs | A disk read per panel, and two panels able to disagree about one scan |
| Defer the viewmodel again, to M6 | The condition ADR-0055 set was met by M5-T05; deferring it again would make the refusal permanent by habit |

## Compliance

- `tests/gui/test_session_viewmodel.py` drives every path with **no widget constructed**.
- `tests/unit/test_import_graph.py::test_no_panel_holds_the_composition_root` fails if a module
  under `gui/panels/` imports `nanoscope.app`.
- `tests/unit/test_import_graph.py::test_the_gui_does_not_reach_past_the_application_layer` fails if
  any module under `gui/` imports `core.science`, `infrastructure` or torch — **M5's fifth exit
  criterion**, which had no check until this task.
- `tests/gui/test_main_window.py::TestTheWindowIsWiredThroughTheSession` asserts a selection reaches
  the viewer without the explorer knowing the viewer exists, and that a display refusal is a status
  line rather than a dialog.

## References

- ADR-0055 §4 — the refusal this pays, and the condition it set
- ADR-0056 — the viewer, and the "status line, not a dialog" rule generalised in §5
- ADR-0043 — the worker-thread listener §7 exists for
- ADR-0041 — *a use case earns its place or is not written*, applied here to a layer
- `docs/Architecture.md` §2.3, §3.1, §3.2
