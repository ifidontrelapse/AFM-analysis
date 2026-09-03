# CURRENT TASK

**ID:** `M8-T04`
**Title:** Training-run persistence: config, metrics, artifacts, provenance
**Milestone:** M8 — Training module, fourth task
**Defect:** — · **ADR:** **ADR-0084** (to be written)
**Branch:** `feat/m8-training`
**Status:** **planning 2026-09-03.** Not started.

---

## Why this task is fourth

M8-T01 wrote the port and said, in its own docstring, what this task is for: *"a `Job` is
in-process and **dies with the process**; a training run has to be findable after a restart
(M8-T04)."* M8-T03 produced the first model this project has ever made and left the other half of
ADR-0006's compliance clause open in its own scope list: *"Registering the produced model — M8-T04
persists the run and registers the `ModelDescriptor`."*

So today nothing survives the process. `LocalTrainingProvider` keeps its runs in a dict; close the
application and a six-hour run is a `best.pt` under `models/` with nothing saying what produced it,
what it trained on, or what it scored. **The weights are on disk and the provenance is in RAM.**

---

## What was measured before planning

**1. A run cancelled before its job starts stays `pending` for ever.** Not read — run, with a
one-worker `JobRunner` occupied by another job:

```
run status after an immediate cancel: pending
run status one second later:          pending
```

`JobRunner` drops a job whose future cancels before it starts (ADR-0043, and `Job.cancel` says so),
so `_train` never runs and nothing publishes. The provider's own `_cancel_pending` set closes the
window between *submit returned* and *there is a handle*, and this is the window **after** it: the
handle exists, the cancel lands, and the body never does. The contract suite does not catch it —
`test_start_returns_before_the_training_is_over` cancels and never waits.

It is this task's business because it is exactly the run the record cannot describe: **a snapshot
that never reaches a terminal state is a `pending` row no restart can resolve**, and to an operator
it is ADR-0043's own failure mode, a cancel button that appears to do nothing.

**2. `values` is a reserved word in SQLite.** `CREATE TABLE a(values TEXT)` is a syntax error on
3.50.4. The column is `metrics`.

**3. What a record actually weighs.** A run is one row plus one row per epoch, each carrying at most
six named floats — `METRIC_BLOCKS` is `train_loss` plus five. Three hundred epochs is three hundred
short rows. Nothing here is large, and nothing here is recomputable: a run cannot be re-run to get
its history back, which is ADR-0044's test for *table, not file*.

**4. Where the vocabulary lives.** `METRIC_BLOCKS` is declared once, in `core`, and ADR-0080 named
its own next change: *"`METRIC_BLOCKS` will need a new block the first time a trainer reports
something real that is not in it — a learning rate, a per-class mAP."* A column per metric would
copy that vocabulary into the migration list, where the copy needs a schema version to change.

---

## The decisions

**1. The project is the memory; the provider only knows the live run.**

`ProjectRepository` gains three methods — `save_training_run`, `get_training_run`,
`list_training_runs` — and the schema gains `training_runs` and `training_epochs` (**v9**).
`TrainingProvider.status` stays what it is: the live view, from the object that started the run. A
restart asks the project, not the provider, which is the split ADR-0080 §2 drew when it refused to
make a run a `Job`.

**2. An epoch is a row and its numbers are JSON in it.**

`training_epochs(run_id, epoch, metrics)`, primary key `(run_id, epoch)`. Not six nullable columns:
ADR-0080 §4 made a block *present in full or absent in full*, and a wide row is the shape it refused
one layer up — plus it puts `core`'s vocabulary in the schema, so ADR-0080's predicted new block
would need a migration to store a number the entity already allows. JSON in a column is the shape
`annotations.points` (ADR-0072) and `settings.value` (ADR-0047) already use here, and for the same
reason: read and written whole, never queried by key. `EpochMetrics.__post_init__` validates on the
way back in, so a row this application cannot name fails at the read rather than becoming a chart.

**3. The run row is columns, and only the shapes are JSON.**

Status, dataset root, counts, every `TrainingConfig` field, the paths and the two timestamps are
columns — that is what makes *"which runs used this dataset"* a query rather than a scan. The two
that are not scalars are: `classes`, a JSON array in index order (`ModelDescriptor.class_map` is
built from it — `DatasetSpec` says so), and `device`, a JSON object, because a resolved `Device` is
three fields that are absent together and three nullable columns can disagree about whether a run
ever started.

**4. Persistence is a use case, and it hangs off the listener the port already has.**

`application/use_cases/training.py::start_training` starts a run through the provider with a
listener of its own, saves every snapshot, and forwards to the caller's listener. No new thread
policy, no repository handed to a provider: `infrastructure/training/` stays unable to name storage,
which is the layering, and the use case is where the policy of *what to keep* belongs (ADR-0041).

**The snapshot `start` returns is not saved.** It would be a write from the calling thread racing
the worker's first callback, and the loser is whichever lands last — a `pending` row over a
`succeeded` one. The port already promises ordered snapshots (M8-T01's contract asserts it), and the
first one arrives in milliseconds, so the listener is the only writer.

**5. A succeeded run registers the model, in the same act.**

That is ADR-0006's compliance clause — *the trained model is registered as a `ModelDescriptor`* —
and M8-T03's stated remainder. The caller names it (`model_id`, and the framework whose provider it
picked), and everything else comes off the run: `path` is `weights_path`, `input_size_px` is
`config.image_size_px`, `class_map` is `dataset.classes` in index order, and `provenance` is a
sentence naming the dataset, the counts, the epochs, the base model and the device. **Only
`SUCCEEDED` registers**, and only with a `weights_path` — a model row pointing at a file a cancelled
run never wrote is the disagreement ADR-0080 §5 removed by refusing `collect_artifacts()`.

**6. A checksum of a file this project just produced is computed, not asked for.**

`register_model` fills `sha256` when the caller gave none and the file is there. ADR-0050 left it
`None` *"if nobody computed it"*, and `application` may not touch the filesystem — but `add_image`
has computed its own checksum since M4-T03 for ADR-0040's reason: *a checksum a caller passes in can
describe a different file.* Same rule, same layer, one `if`.

**7. A run interrupted by a crash stays `running` in the record.**

It is what was true when the process died. Marking it `failed` on read invents an outcome nobody
observed — the substitution ADR-0025 and ADR-0033 removed elsewhere — and there is no `resume`
(ADR-0080's named negative), so nothing can honestly finish it. M8-T05 is what shows a stored run
whose id no live provider knows; this task states the consequence rather than papering it.

---

## Scope

**In scope**

1. Schema **v9** — `training_runs`, `training_epochs`; `tests/integration/schema_history.py` extended
2. `ProjectRepository` port + `SqliteProjectRepository`: `save_training_run`, `get_training_run`,
   `list_training_runs`
3. `application/use_cases/training.py` — `start_training`, and the descriptor a run produces
4. `register_model` computes a missing checksum
5. **The defect in §1 of the measurements** — a run cancelled before its job starts reaches
   `CANCELLED` — with **one new assertion in the contract suite**, which the fake already satisfies
6. **ADR-0084** + the ADR index

**Out of scope**

- **Any UI** — M8-T05 has the window; nothing is wired into the container (ADR-0041, seventh
  application)
- **Resumption** — still needs more than a stored path, and ADR-0080 named it
- **Backfilling old runs** — there are none; nothing has ever been persisted
- **Deleting a run's record or its weights** — no caller, and the deletion policy for a 137 MB
  artifact is an operator's decision (M8-T06)

---

## Definition of done

- [ ] A finished run, its config, its every epoch and its device survive closing and reopening the project
- [ ] A succeeded run leaves a registered `ModelDescriptor` pointing at the weights it produced
- [ ] A run with nothing held out comes back with **no** `validation` block, on every epoch
- [ ] A run cancelled the instant it starts reaches a terminal state, and the contract says so
- [ ] ADR-0084 + the ADR index
- [ ] `make check` green, golden byte-identical
- [ ] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [ ] Commit: `M8-T04: a run the project remembers, and the model it produced`
