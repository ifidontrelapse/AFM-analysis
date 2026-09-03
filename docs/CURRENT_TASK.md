# CURRENT TASK

**ID:** `M8-T05`
**Title:** Training UI: configuration, live metrics, cancellation
**Milestone:** M8 — Training module, fifth task
**Defect:** — · **ADR:** **ADR-0085** (to be written)
**Branch:** `feat/m8-training`
**Status:** **planning 2026-09-03.** Not started.

---

## Why this task is fifth

Four tasks built a training module nothing calls. M8-T01 wrote the port, M8-T02 the dataset,
M8-T03 the trainer, M8-T04 the record — and each one closed on the same sentence: *not wired into
the composition root (ADR-0041); the caller arrives with M8-T05.* This is that caller, and it is
the first task in this milestone an operator can see.

Three debts were named by the tasks that created them and come due here:

- **M8-T02:** *"building preprocesses every scan and is not a job — the runner exists, the caller
  that needs it arrives in M8-T05."*
- **M8-T04 §8:** a run interrupted by a crash stays `running` in the record, and *"M8-T05 is what
  shows a stored run whose id no live provider knows."*
- **ADR-0041's seventh application:** the container constructs no `TrainingProvider`.

---

## What was measured before planning

**1. Closing the application during a run blocks for the whole run, and never asks it to stop.**
`Nanoscope.close()` calls `jobs.shutdown(wait=True)`. Run, with a six-second checkpointed job:

```
close() took 6.01 s with a running job
job state after close: succeeded   cancellation asked: False
```

The window is gone by then — `closeEvent` has run — so this is a process with no UI, no progress
and no cancel button, for as long as the work takes. Today the longest job is an import of a folder;
**this task makes the longest job six hours.**

**And `wait=False` does not fix it**, which was worth running rather than assuming:

```
shutdown(wait=False) returned after 0.00 s
process exited after 5.06 s total
```

`concurrent.futures` joins its non-daemon threads at interpreter exit, so `wait=False` moves the
hang from `close()` to exit and buys nothing. The honest fix is to **ask, and cancel** — which
lands at an epoch boundary, ADR-0043's own promise.

**2. Building the dataset costs 627–651 ms per scan.** Synthetic 512×512 AFM scans, one box each:

| Scans | `build_dataset` | Per scan |
|---|---|---|
| 10 | 6.51 s | 651 ms |
| 40 | 25.10 s | 627 ms |

Forty scans is 25 seconds of a frozen window if this is called where the button is. It is a job,
and M8-T02 said so when it wrote it.

**3. `is_busy` gates ten actions.** `MainWindow._update_actions`: New, Open, Close, Import, Remove,
four exports, Import annotations, Undo and Redo. A training run in that slot is an application an
operator cannot annotate, undo or export in for six hours — and the docstring says why the gate
exists: *"`close_project()` closes the SQLite connection the worker thread is using."*

**4. Nothing supplies a default `base_model`, and the window may not invent one.**
`TrainingConfig.base_model` is required, every caller today is a test that types one, and
`TestNoDetectorNameLivesInTheGui` greps `nanoscope/gui/` for the string `yolo` and fails on it
(PROJECT_RULES §2.5, D-19's lesson).

---

## The decisions

**1. A window, not a tenth dock.**

Nine docks exist, and `apply_default_layout`'s docstring carries the measurement that made the
right-hand group tabbed: *811 px of minimum height against 1 785 for the same five untabbed.* A
tenth panel competes for that space to ask a question that is not about the selected scan — every
dock on the right answers *what about this image?*, and training is a project-level act.

So: a **modeless** `QDialog` from the menu, closable while the run continues. Modeless because M5's
third exit criterion rules out the alternative in as many words — *a long-running job shows progress
and can be cancelled **without freezing the UI*** — and a modal training window is the frozen
application it is reporting on. Closing it does not stop the run: the run is in the project
(M8-T04), and the status bar already shows the job without knowing what training is, which is what
M8-T03 routed progress through `JobContext.report` for.

**2. Training does not take the session's job slot. It gets its own.**

`is_busy` stays exactly what it is — one short job owning the project's connection — and
`is_training` is a second question. Only the project-lifecycle actions (New, Open, Close) read
both; annotating, undoing and exporting stay available while a model trains, because a repository
write from a worker and one from the main thread are already serialised by one lock, which
`_serialised`'s docstring says is for exactly this.

The cost is stated rather than hidden: `max_workers` is 2, so **with a run going the application has
one worker instead of two** — an import and an analysis no longer overlap.

**3. The dataset build is a job, and it is the run's first step.**

One button, one job: build the dataset, then start the run from inside it. `build_dataset` gains
`progress: JobContext | None = None` — **the parameter `import_images` already has, by that name**,
rather than a second way of saying the same thing. Twenty-five seconds is not a spinner's worth of
silence, and it is the only part of this an operator can usefully cancel.

The job ends when training *starts*, which is honest: the build is a job and the run is a run, and
they have different lifetimes, different terminal states and different records.

**4. Live metrics are a table, and its columns are `METRIC_BLOCKS`'.**

Not a chart. Six named scalars over an axis of epochs is a legend and an axis choice before it is
information, and `EpochMetrics` is already the shape of a row. The vocabulary is read off
`METRIC_BLOCKS` rather than typed here, for ADR-0031's reason one layer on: a widget with its own
column list is the copy that drifts, and ADR-0080 has already promised the vocabulary will grow.

A run with nothing held out shows no validation columns, because there is no validation block —
ADR-0082's distinction, visible.

**5. The starting point is a choice the application offers, not a name the window types.**

`application/use_cases/training.py` gains `starting_points(repository)`: the framework's own fresh
start, plus every registered `DETECT` model, **by the id its operator gave it** (ADR-0050). The
window renders what it is handed. That is where `capabilities.py` already keeps detector names, and
it is the only place this one can live — measurement 4.

**6. A stored run no live provider knows is shown as interrupted, and is not called failed.**

M8-T04 §8 decided the record; this decides the sentence. The history lists what the project stored
(`list_training_runs`), and a `RUNNING` row whose id `TrainingProvider.status` refuses is labelled
*interrupted — this process is not running it*. Not `failed`: nobody observed a failure, which is
the substitution ADR-0025 and ADR-0033 removed elsewhere, and there is no `resume` to offer.

**7. Closing the application during a run asks first.**

The measured defect, fixed where an operator is: `closeEvent` asks when a run is live, and closing
cancels it. Cancelling lands at an epoch boundary — ADR-0043 §3's honest limit, and the same
sentence `JobStatus` already shows — so the window says *stopping*, not *stopped*.

Fixed in the window rather than the container because the container has no operator to ask, and
`wait=False` there was measured to buy nothing.

---

## Scope

**In scope**

1. `Nanoscope` constructs a `LocalTrainingProvider` for the open project (ADR-0041, seventh)
2. `SessionViewModel`: `train`, `cancel_training`, `training_runs`, `starting_points`,
   `is_training`, and a `training_changed` signal marshalled off the worker thread (ADR-0058)
3. `build_dataset(progress=...)` — the M8-T02 debt, with `import_images`' parameter
4. `application/use_cases/training.py::starting_points`
5. `nanoscope/gui/dialogs/training.py` — configuration, the live epoch table, the history, cancel
6. `MainWindow`: the menu item, `is_training` in `_update_actions`, and `closeEvent` asking
7. **ADR-0085** + the ADR index

**Out of scope**

- **Model management UI** — import, register, activate, compare is M8-T06
- **A chart** — decision 4; the table is the deliverable, and a chart is an addition to it
- **Resumption** — still ADR-0080's named negative, and still nothing can honestly finish a run
- **Deleting a run or its weights** — M8-T06, and unchanged from M8-T04's reasoning
- **The remote provider** — M8-T07; this window talks to the port, so it does not care

---

## Definition of done

- [ ] Annotations → dataset → weights → a registered model, **from the window**, with no code change
- [ ] The build reports progress and can be cancelled; the run reports epochs and can be cancelled
- [ ] A run with nothing held out shows no validation columns
- [ ] A stored run nobody is running reads as interrupted, not as failed
- [ ] Closing the window during a run asks, and closing cancels it
- [ ] The application stays usable while a model trains, and the project cannot be closed under it
- [ ] ADR-0085 + the ADR index
- [ ] `make check` green, golden byte-identical
- [ ] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [ ] Commit: `M8-T05: the window that turns annotations into a model`
