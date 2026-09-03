# ADR-0085 — The window that starts a run is not the one that waits for it

- **Status:** Accepted
- **Date:** 2026-09-03
- **Deciders:** operator + agent (M8-T05)
- **Affects:** `gui/dialogs/training.py`, `gui/main_window.py`, `gui/viewmodels/session.py`,
  `app/container.py`, `application/use_cases/` · M8

## Context

Four tasks built a training module nothing called. M8-T01 wrote the port, M8-T02 the dataset,
M8-T03 the trainer, M8-T04 the record — and every one of them ended on the same sentence: *not wired
into the composition root (ADR-0041); the caller arrives with M8-T05.* This is that caller, and it
is the first task in this milestone an operator can see.

Three debts were named by the tasks that created them and come due here:

| Named by | What it said |
|---|---|
| M8-T02 | *"building preprocesses every scan and is not a job — the caller that needs it arrives in M8-T05"* |
| M8-T04 §8 | *"M8-T05 is what shows a stored run whose id no live provider knows"* |
| ADR-0041 | the composition root constructs no `TrainingProvider` |

Four things were measured before anything was written.

**1. Closing the application during a run blocks for the whole run, and never asks it to stop.**
`Nanoscope.close()` calls `jobs.shutdown(wait=True)`. With a six-second checkpointed job:

```
close() took 6.01 s with a running job
job state after close: succeeded   cancellation asked: False
```

The window is gone by then — `closeEvent` has already run — so this is a process with no UI, no
progress and no cancel button, for as long as the work takes. Today's longest job is an import of a
folder. **This task makes the longest job six hours.**

And `wait=False` does not fix it, which was worth running rather than assuming:

```
shutdown(wait=False) returned after 0.00 s
process exited after 5.06 s total
```

`concurrent.futures` joins its non-daemon threads at interpreter exit, so `wait=False` moves the
hang from `close()` to exit and buys nothing.

**2. Building the dataset costs 627–651 ms per scan** — 6.51 s for ten 512×512 scans, 25.10 s for
forty. Twenty-five seconds of a window that has stopped repainting, if it happens where the button
is.

**3. `is_busy` gates ten actions.** New, Open, Close, Import, Remove, four exports, Import
annotations, Undo, Redo — and `_update_actions`' docstring says why: *"`close_project()` closes the
SQLite connection the worker thread is using."*

**4. Nothing supplies a default `base_model`, and the window may not invent one.** The field is
required, every caller today is a test that types one, and `TestNoDetectorNameLivesInTheGui` greps
`nanoscope/gui/` for the string `yolo` and fails on it (PROJECT_RULES §2.5, D-19's lesson).

## Decision

### 1. A window, not a tenth dock, and modeless

Nine docks exist, and `apply_default_layout` carries the measurement that made the right-hand group
tabbed: *811 px of minimum height against 1 785 for the same five untabbed*. Every dock on the right
answers *what about this scan?*; training is a project-level act, and a tenth panel would compete for
that space to ask a question that is not about the selected image.

**Modeless** because M5's third exit criterion rules out the alternative in as many words — *a
long-running job shows progress and can be cancelled without freezing the UI* — and a modal window
over a six-hour run is the frozen application it is reporting on. Closing it does not stop the run:
the run is in the project (ADR-0084), and the status bar already shows the job **without knowing what
training is**, which is what M8-T03 routed progress through `JobContext.report` for.

The window is kept rather than rebuilt per press, so a second press raises the one that is watching
the run instead of opening a second beside it.

### 2. Training does not take the session's job slot; it gets its own question

`is_busy` stays exactly what it is — one short job owning the project's connection — and
`is_training` is a second property. Only the three project-lifecycle actions read both. Annotating,
undoing and exporting stay available while a model trains, because a repository write from a worker
and one from the main thread are already serialised by one lock, which `_serialised`'s docstring says
is for exactly this case.

The cost is stated rather than hidden: `max_workers` is 2, so **with a run going the application has
one worker instead of two**, and an import no longer overlaps an analysis.

### 3. The two gates meet at the end of the build job, and the handoff is a hole that had to be closed

The build is a job (`is_busy`); the run is not (`is_training`). Between them was a window nothing
covered, and it is **not** a test artefact: `start_training` deliberately does not write the snapshot
`start` returns (ADR-0084 §4), so nothing knew a run existed until its **first epoch callback** —
milliseconds for the fake, *minutes* for a real trainer. In that window `is_training` said no, Stop
was disabled, and **Close Project was enabled** — which closes the SQLite connection the run is
writing through.

So the session adopts the returned snapshot as **local state only**, and only when nothing about that
run has arrived yet. ADR-0084 §4's rule, applied to memory instead of the database: the loser must
never be the older snapshot.

### 4. The dataset build is a job, and a cancelled build produces nothing

`build_dataset` gains `progress: JobContext | None = None` — **the parameter `import_images` already
has, by that name**, rather than a second way of saying the same thing.

It **raises where `import_images` breaks**, and the difference is what a partial result means. A
stopped import is a project with fewer scans in it: a state an operator can see, and one the report
describes. A stopped build is a training set quietly missing whatever came after the button, and
nothing downstream could tell — so the job ends `CANCELLED` and no run starts. The half-written
directory is under `cache/`, which is deletable by definition (ADR-0081).

### 5. Live metrics are a table, and a block nothing measured is not a column of blanks

Not a chart: six named scalars over an axis of epochs is a legend and an axis choice before it is
information, and `EpochMetrics` is already the shape of a row. The columns are read off
`METRIC_BLOCKS` rather than typed here — ADR-0031's rule one layer on, and ADR-0080 has already
promised the vocabulary will grow.

**A metric no epoch of this run reported is a hidden column, not an empty one.** ADR-0082 made
`validation` mean *a held-out set existed*, so a run with nothing held out has no precision to show
and no honest place to show one; a `precision` header over five blank cells is a question an operator
spends the run wondering about, and a `0.000` there would be a score nobody measured.

### 6. The starting point is a choice the application offers, not a name the window types

`starting_points(repository)` returns the framework's own fresh start plus every registered `DETECT`
model, **by the id its operator gave it** (ADR-0050: *an operator names their model*) — a combo box
listing checkpoint filenames is one nobody can choose from. Segmentation models are not offered:
they are imported rather than trained here (M8-T06), and starting a detector from one is a run that
fails four seconds in with a framework's error message instead of this layer's sentence.

The framework's name lives in `application`, which is where `capabilities.py` already keeps detector
names, and `TrainingConfig.base_model` is *"passed through, never interpreted"* — so this is a string
on its way to a provider, not this layer deciding about a framework.

### 7. A stored run no live provider knows is *interrupted*, and is not called failed

M8-T04 §8 decided the record — a crashed run stays `running`, because that is what was true when the
process died. This decides the sentence. The history lists what the project stored, and a `running`
row whose id `TrainingProvider.status` refuses is labelled **interrupted**, with a tooltip saying
that nothing here can honestly say whether it finished. Not *failed*: nobody observed a failure,
which is the substitution ADR-0025 removed for scales and ADR-0033 for heights, and there is no
`resume` to offer either (ADR-0080's named negative).

### 8. Closing the application during a run asks first

The measured defect, fixed where there is an operator to ask. `closeEvent` asks when a run is live,
and closing cancels it — which lands at the next epoch boundary, which is all ADR-0043 §3 ever
promised, and the question says so. What was trained by then is kept on disk and recorded in the
project; **no model is registered**, because a cancelled run has no weights to register (ADR-0084 §5).

Not fixed in the container: it has no operator to ask, and `wait=False` there was measured to buy
nothing.

## Consequences

**Positive**

- **M8's first exit criterion is met**: annotations → dataset → trained weights → a registered
  `ModelDescriptor`, without leaving the application, and a test presses the button.
- Four tasks of unwired code have a caller. `app.training` is also the seam M8-T07's remote provider
  arrives through, and the GUI tests already use it as one.
- An operator can work while a model trains, and cannot close the project out from under it.
- The M8-T02 debt is paid where it was named, with the parameter that already existed.
- A cancel pressed on the way out of the application does something, and says what it costs.

**Negative**

- **With a run going the application has one worker instead of two.** `max_workers` is 2 and a
  training job holds one for hours. Not raised speculatively: the number is there to be set by the
  composition root *when there is something to measure* (ADR-0043's own words).
- **The window is not the only place a run can be watched.** The status bar shows the build job, the
  dialog shows the run, and the two are different objects with different lifetimes. That is honest
  and it is two places to look.
- **No chart.** Reading a 300-epoch run in the table means scrolling. The upgrade is a painted plot
  beside it (the statistics panel's shape), not a rewrite.
- **Closing during a run can still take an epoch.** The question says so rather than pretending the
  process exits at once — but an operator who says *close* watches a window that is already gone
  until the boundary comes.
- `Nanoscope.open` now constructs one more object. It is three arguments and no I/O; the weights are
  loaded by the run, not by the provider.

**Neutral**

- The training dialog is this application's first modeless one, and `dialogs/`' docstring — *"modal
  questions the window has to ask"* — now names the exception.
- The metric table's columns are derived, so ADR-0080's predicted new block appears on screen the
  day it is added to `core`.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A tenth dock | Competes for the space `apply_default_layout` measured, to ask a question that is not about the selected scan |
| A modal dialog | The frozen window M5's third exit criterion exists to prevent, with a progress bar on it |
| Training takes `is_busy` | Ten actions disabled for six hours, Undo among them: a training appliance, not a training feature |
| Nothing gates the project while a run goes | `close_project()` closes the connection the run is writing through — the defect `_update_actions` was written against |
| Build the dataset on the main thread | 25.1 s of a window that has stopped repainting, measured |
| A cancelled build returns what it managed | A training set quietly missing scans, and nothing downstream can tell — unlike a partial import, which is a visible state |
| Save the snapshot `start` returns, to close the handoff hole | ADR-0084 §4 refused that write for a reason that has not changed; adopting it as local state has the same effect and none of the race |
| A chart of the six metrics | A legend and an axis choice before it is information, over a vocabulary ADR-0080 says will grow |
| Blank cells for a block nothing measured | A `precision` header over empty cells is a question an operator spends the run wondering about; a `0.000` is a score nobody computed |
| The window names the starting weights | PROJECT_RULES §2.5, enforced by a grep, and D-19 is what the other outcome looked like |
| `shutdown(wait=False)` in the container | Measured: returns in 0.00 s, and the process still takes the full run to exit |
| Close without asking, cancelling silently | Six hours discarded by a window-close nobody was warned about |
| Leave the close-hang for M9 | This task is what makes the longest job hours instead of seconds |

## Compliance

- `tests/gui/test_training_dialog.py` — 24 tests: the whole path from the window registers a model
  with its class map and checksum; the run is in the project afterwards; a run with nothing held out
  shows **no** validation columns and one that held something out shows them; the columns *are*
  `METRIC_NAMES`; a stored `running` run nobody is running reads as **interrupted**; the run is live
  the moment the build ends; a cancelled build starts no run; `is_busy` and `is_training` are
  different questions, with the three lifecycle actions disabled and the exports enabled; closing
  asks, saying no keeps the window, saying yes cancels the run, and with no run it does not ask.
- `tests/integration/test_dataset_builder.py` — the build reports one step per scan with a named
  message, a cancelled build raises and produces nothing, and `progress=None` still builds.
- `tests/integration/test_training_history.py` — `starting_points` offers a fresh start alone, then
  the project's own detectors by their operator-given ids, and never a segmentation model.
- The golden is byte-identical. Nothing in this task is on a numerical path.

## References

- ADR-0080 / ADR-0081 / ADR-0082 / ADR-0084 — the port, the dataset, the trainer and the record this window calls
- ADR-0006 — the seam, and the compliance clause §1 and §6 close from a window
- ADR-0043 — jobs, checkpoints, and what a cancel can honestly promise
- ADR-0041 — *a use case earns its place*; the seventh application, and the last one M8 owed
- ADR-0050 — an operator names their model, which is why §6 offers ids and not filenames
- ADR-0057 / ADR-0058 — one viewmodel, and the queued signal that carries a worker's snapshot
- ADR-0031 — one quantity, one name, and a block present in full or absent in full
