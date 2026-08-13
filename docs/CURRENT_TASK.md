# CURRENT TASK

**ID:** `M5-T06`
**Title:** The viewmodel holds the session, and a widget emits intent
**Milestone:** M5 — GUI shell, sixth task
**Defect:** — · **ADR:** **ADR-0057**
**Branch:** `feat/m5-gui-shell`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

ADR-0055 §4 declined a viewmodel **with a condition attached**:

> *When M5-T05's viewer needs the same selection, there will be two consumers and a reason.*

M5-T05 shipped that viewer. The condition is met, so this task is the payment — and the three
things it has to fix are visible in the code as it stands:

1. **Both panels talk to the container.** `ProjectExplorer(app)` reads `app.repository`, calls the
   `open_project` use case and `remove_image`; `ImageViewer(app)` calls `load_for_display`. Which
   image to load when a row is clicked is a *decision*, and Architecture §2.3 says a widget does not
   make them.
2. **A widget is wired to another widget.** `explorer.image_selected.connect(viewer.show_image)` in
   `MainWindow._build_docks`. With two panels that is one line; the Properties dock makes it three,
   and every panel after that pays the same tax — n panels, n² connections, and no single place that
   says what the session currently is.
3. **The Properties dock has no source of truth to fill it from.** It needs the image the viewer has
   *already loaded*. Without somewhere to hold it, the honest implementations are "read the file a
   second time" or "ask the viewer widget", and both are worse than the abstraction.

M5-T07 adds a third reason it does not get to skip: ADR-0043 says a job's listener fires **on the
worker thread**, three times over, and the GUI must marshal. A `QObject` with signals is the thing
Qt marshals across threads; a widget method called from a worker is a crash.

---

## The decisions this task has to make

**1. How many viewmodels? One.**

Not one per view. What more than one panel needs is the *session*: the open project, which image is
selected, and the loaded array. Per-view state that nobody else can want — the viewer's colormap,
its full-range checkbox, its zoom — stays in the widget, because a viewmodel per view would be four
classes forwarding to one.

This is ADR-0041's rule at a different altitude: *a layer earns its place or is not written*. What
earns it here is shared state and cross-thread signalling, not symmetry with a diagram.

**2. What do the panels receive? The viewmodel, and not the container.**

`ProjectExplorer(session)`, `ImageViewer(session)`, `PropertiesPanel()`. After this task no module
under `gui/panels/` imports `nanoscope.app` at all — which is checkable, so it gets checked rather
than reviewed (M5-T03's rule: the rule and its enforcement ship together, or only the rule does).

**3. Who talks to whom? Widgets emit intent; the viewmodel emits state; no widget listens to a
widget.**

The explorer emits "the operator picked row N" → the viewmodel loads → the viewmodel emits
`image_changed` → the viewer draws it and the properties panel fills. One connection per panel,
made in one place, and a new panel connects to the viewmodel rather than to every panel before it.

**4. Where do dialogs live? In the widget.**

The viewmodel holds no `QWidget` and opens no `QMessageBox`. A viewmodel that pops a modal box is
one that cannot be tested without a window, and the confirmation ADR-0055 built is a *view*
decision about how to ask. The viewmodel supplies the count and performs the removal; the explorer
asks the question.

**5. What does a refusal do?** It becomes one `failed(str)` signal.

The window puts it in the status bar. A dialog appears only where the operator pressed a button
labelled with the action — opening a project — which is ADR-0056's rule stated once instead of per
panel: *the operator clicked a row, not a button labelled "load"*.

**6. The Properties dock is filled, from the image already in memory.**

Name, modality, size in px, size in nm (or "scale unknown"), pixel size, dtype, and the value range
— reusing `value_range(image, full=True)` rather than a second min/max written in a widget. Run
properties wait for M6, because there are no runs in this GUI yet, and the dock will say so.

---

## Scope

**In scope**

1. `gui/viewmodels/session.py` — `SessionViewModel(QObject)`: `project_changed`, `image_changed`,
   `failed`; `open_project`, `close_project`, `refresh`, `select_image`, `annotation_count`,
   `remove_image`
2. `gui/panels/properties.py` — the dock M5-T02 promised for this task
3. `gui/panels/project_explorer.py`, `gui/panels/viewer.py` — take the viewmodel, keep their widgets
4. `gui/main_window.py` — constructs the viewmodel, makes every connection, keeps the dialogs
5. **ADR-0057** — one viewmodel, intent up / state down, no dialogs below the widget
6. Tests: the viewmodel driven with **no widget at all**; the panels reacting to its signals; the
   guard that no panel imports the container

**Out of scope**

- **Marshalling a job's listener** (ADR-0043) — M5-T07 owns the runner; this task only makes the
  object it will marshal onto exist
- **Detections and annotations drawn over the image** — M6's editor
- **A per-view viewmodel for the viewer's own controls** — decision 1; the trigger is a second
  consumer of the colormap, and there is none

---

## Definition of done

- [x] One `SessionViewModel`, exercised in tests without constructing a widget
- [x] No module under `gui/panels/` imports `nanoscope.app`, proven by a test
- [x] No widget connects to another widget's signal
- [x] The Properties dock shows the loaded image, and no file is read twice
- [x] ADR-0057 + the ADR index
- [x] `make check` green — 972 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `Roadmap.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M5-T06: the viewmodel holds the session, and a widget emits intent`

---

## What it turned up

**The new guard failed on the layer it exists to permit.** `name.startswith("nanoscope.app")` also
matches `nanoscope.application` — so the check that panels do not hold the composition root
rejected `nanoscope.application.use_cases.display`, which is exactly what a panel is *supposed* to
import. Caught by running it; the rest of the file had used `== bad or startswith(bad + ".")` since
M2-T09, and this is why.

**M5's fifth exit criterion had no owner and no check.** *"Lint rule proves no `gui/` module
imports `core.science` or `infrastructure`"* was true by habit and unenforced. No task in the M5
list claimed it, and this task was already writing the file that enforces layering in `gui/`, so it
was added here — as a test, which is the form every other rule in this project takes.

**Opening a second project kept the first one's selection.** Image ids are per-project, so image 3
of the old project is not image 3 of the new one, and a panel would have gone on showing a stranger
until the next click. The selection is cleared where the project changes rather than in each panel.

**"Modality: afm".** Found by rendering a phantom into a window and reading the panel, not by a
test — the enum's value is a storage token, and three instrument names that are acronyms everywhere
else read as typos. M5-T05's lesson, repeated: two minutes of looking is worth a test suite that
passes.
