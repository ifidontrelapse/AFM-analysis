# ADR-0084 — A training run is remembered by the project, not by the provider

- **Status:** Accepted
- **Date:** 2026-09-03
- **Deciders:** operator + agent (M8-T04)
- **Affects:** `infrastructure/storage/`, `core/ports/project_repository.py`,
  `application/use_cases/training.py`, `infrastructure/training/local.py` · M8

## Context

M8-T01 wrote this task's reason into the entity it defined: *"a `Job` is in-process and **dies with
the process**; a training run has to be findable after a restart (M8-T04) and may be executing on a
machine this application did not start (M8-T07)."* M8-T03 then produced the first model this project
has ever made, and left the other half of ADR-0006's compliance clause — *the trained model is
registered as a `ModelDescriptor`* — in its own out-of-scope list, naming this task.

So the state before this one: `LocalTrainingProvider` keeps its runs in a dict, and closing the
application turns six hours of training into a `best.pt` under `models/` with nothing saying what
produced it, what it trained on, how it scored or which device it ran on. **The weights are on disk
and the provenance is in RAM.**

Three facts were measured before anything was written.

| What | Measured |
|---|---|
| A run cancelled before its job starts | Stayed `pending` immediately, a second later, and for ever |
| `values` as a column name | `CREATE TABLE a(values TEXT)` — syntax error, SQLite 3.50.4 |
| What a record weighs | One row, plus one row per epoch carrying at most six floats — `METRIC_BLOCKS` is `train_loss` and five |

The first is a defect, and it is this task's because it is the run the record cannot describe:
`JobRunner` **drops** a job cancelled before it starts (ADR-0043, and `Job.cancel` says so), so
`_train` never runs and nothing publishes. A snapshot that never reaches a terminal state is a row
no restart can resolve, and on screen it is ADR-0043's own failure mode — a cancel button that
appears to do nothing. The contract suite did not catch it: its first test cancels and never waits.

## Decision

### 1. The project is the memory; the provider only knows the live run

`ProjectRepository` gains `save_training_run`, `get_training_run` and `list_training_runs`, and the
schema gains `training_runs` and `training_epochs` (**v9**). `TrainingProvider.status` stays exactly
what it was — the live view, from the object that started the run, in one process. A restart asks
the project.

That is the split ADR-0080 §2 drew when it refused to make a run a `Job` and duplicated five state
names to do it. This ADR is where the duplication pays: the enum in `core` is the one that gets
stored, because it is the one that describes work outliving a thread pool.

### 2. An epoch is a row, and its numbers are JSON in it

`training_epochs(run_id, epoch, metrics)`, primary key `(run_id, epoch)`.

Not a column per metric. ADR-0080 §4 declared the vocabulary **once**, in `core`, and named its own
next change: *"`METRIC_BLOCKS` will need a new block the first time a trainer reports something real
that is not in it — a learning rate, a per-class mAP."* A column list is that vocabulary copied into
the migration file, where a copy needs a schema version to change and can disagree in the meantime.
It is also the wide-record-with-holes shape ADR-0031 refused one layer up: a block is present in
full or absent in full, and six nullable columns invite half of one.

JSON in a column is the shape `annotations.points` (ADR-0072) and `settings.value` (ADR-0047)
already use here, for the reason that applies again: read and written **whole**, never queried by
key. And the read goes through `EpochMetrics`, whose constructor refuses an unknown name and a
partial block — so a row this application cannot name fails at the read rather than becoming a
chart.

`metrics`, not `values`: measured above.

### 3. The run row is columns; only the shapes are JSON

Status, dataset root, counts, every `TrainingConfig` field, the paths and the two timestamps are
columns, which is what makes *"which runs used this dataset"* a query. Two fields are not scalars:
`classes`, a JSON array in index order — `DatasetSpec` says a `ModelDescriptor.class_map` is built
from it, and ADR-0081 put the dataset in `cache/`, which is deletable by definition — and `device`,
a JSON object, because a resolved `Device` is three fields that are absent **together**: a run that
never started ran nowhere, and three nullable columns can disagree about that.

`config.device` and `run.device` are both stored, under different names. They are different facts:
one is what was asked for, the other is what the manager resolved, and ADR-0049 is the ADR about
what a fallback costs.

### 4. Persistence is a use case, hanging off the listener the port already has

`start_training` starts a run through the provider with a listener of its own, writes every snapshot
it publishes, and forwards to the caller's listener afterwards.

No repository is handed to a provider: `infrastructure/training/` may not name storage, and *what to
keep* is a policy, which is what a use case is for (ADR-0041). No second thread policy either — the
listener is the seam ADR-0043 already built, and it already runs on the worker.

**The snapshot `start` returns is deliberately not written.** It would be a write from the calling
thread racing the worker's first callback, and the loser is whichever lands last: a `pending` row
over a `succeeded` one. The port promises ordered snapshots and M8-T01's contract asserts it, so the
listener is the only writer, and the first callback arrives in milliseconds.

### 5. A succeeded run registers the model, in the same act

ADR-0006's compliance clause, and M8-T03's stated remainder. The caller names it — an operator names
their model (ADR-0050), and a configuration naming a UUID is one nobody can read — and everything
else comes off the run: `path` is `weights_path`, `input_size_px` is `config.image_size_px`,
`class_map` is `dataset.classes` in index order, `provenance` is a sentence naming the dataset, the
counts, the epochs, the base model and the device.

**Only `SUCCEEDED`, and only with weights.** A model row pointing at a file a cancelled run never
wrote is precisely the disagreement ADR-0080 §5 removed when it refused a `collect_artifacts()`.

`framework` is `ULTRALYTICS` and is not a parameter: the dataset this port consumes is the one
M8-T02 builds, so whatever trains it — here or on another machine (M8-T07) — produces weights
ultralytics loads. A parameter for a value with one possible answer is a question no caller can
answer better.

### 6. A checksum of a file this project just produced is computed, not asked for

`register_model` fills `sha256` when the caller gave none and the weights are there. ADR-0050 left
it `None` *"if nobody computed it"*, and `application` may not touch the filesystem — but this
module has had the rule since M4-T03 and its docstring states it: *a checksum describes the file the
row points at, because it is computed here from that file and never accepted as an argument*
(ADR-0040). A caller who passed one keeps it; nothing re-reads 137 MB to second-guess them, and an
absent file stays `None` rather than becoming a hash of nothing.

### 7. A run cancelled before it trains anything still ends

The measured defect, fixed where it is: the provider passes a **job** listener too, and a job that
reached `CANCELLED` while its run is unfinished publishes `CANCELLED` on the run. Only that case —
a job cancelled while it runs is ended by the body, at the epoch boundary where the checkpoint is.

The contract suite gains its **fifteenth assertion**, which the fake satisfied unchanged: *every run
reaches a terminal state, including the one that never began*. Which terminal state is not asserted;
a provider fast enough to finish first has succeeded, honestly.

### 8. A run interrupted by a crash stays `running` in the record

It is what was true when the process died. Marking it `failed` on read invents an outcome nobody
observed — the substitution ADR-0025 removed for scales and ADR-0033 for heights — and there is no
`resume` (ADR-0080's named negative), so nothing can honestly finish it. M8-T05 is what shows a
stored run whose id no live provider knows.

## Consequences

**Positive**

- A finished run survives the process, which is what M8-T01 wrote the entity for and the first thing
  in this milestone that an operator keeps.
- ADR-0006's compliance clause is closed: annotations → dataset → weights → a registered
  `ModelDescriptor`, without leaving the application. M8's first exit criterion has everything but
  its window.
- A trained model carries a checksum and a provenance sentence, so M8-T06's *compare* has something
  to compare and M8-T08 can say which weights produced a report.
- The metric vocabulary stays in one file. Adding a block is an entity change and a test, not a
  migration.
- A cancel pressed the instant a run starts now does something, and the contract says so for every
  provider — including M8-T07's, before it is written.

**Negative**

- **`training_epochs.metrics` is opaque to SQL.** *"Every run whose best mAP50 beat 0.8"* is a scan
  and a JSON parse rather than an index. Judged the right trade at hundreds of rows per run; the
  upgrade is a generated column, not a rewrite.
- **The whole snapshot is rewritten once an epoch.** A 300-epoch run writes ~45 000 short rows over
  six hours instead of 300. That is nothing against an epoch, and it is what keeps the writer free
  of *which half moved*.
- **A crashed run's row says `running` for ever.** Stated in §8 and left to M8-T05 to display, which
  means the first version of that window has to say something honest about a run nobody is running.
- `register_model` now reads the weights file when no checksum was given. For a 137 MB checkpoint
  that is a fraction of a second, once, and it is skipped when the caller supplies one — but it is a
  new file read on a method that had none.

**Neutral**

- Schema **v9**, and the second migration in this project to add two tables at once.
- The use case is not wired into the composition root. ADR-0041's rule, seventh application: the
  caller arrives with M8-T05's window, and it is five lines.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A column per metric | Copies `core`'s vocabulary into the migration list, and a new block ADR-0080 already predicted would need a schema version to store a number the entity allows today |
| Metrics as a file under `results/`, like a measurement table | ADR-0042 sent that table to a file because its columns vary by producer **and it can be recomputed**. A run cannot be re-run to get its history back — ADR-0044's test, and it says table |
| An EAV row per metric per epoch | Six rows an epoch to store one dict, and every read is a pivot. The block invariant then lives in the reader instead of in `EpochMetrics` |
| The whole `TrainingRun` as one JSON blob | Nothing queryable, and a schema that describes nothing. The project's own tables are columns for this reason |
| Persistence inside `LocalTrainingProvider` | Puts storage in `infrastructure/training/`, duplicates it in the remote provider, and makes the fake need a database to be a fake |
| A `RecordingTrainingProvider` decorator | Elegant, and it cannot carry `model_id`: the port has no place for it, so registration would move somewhere else anyway |
| Save the snapshot `start` returns, too | A calling-thread write racing the worker's first callback, whose loser is a `pending` row over a `succeeded` one |
| Mark a stored `running` run as `failed` when it is read | Invents an outcome nobody observed, and no restart can know whether the process died or the run is still going on another machine |
| Register the model in `save_training_run` | Storage deciding what a model is called. The name is an operator's, and the policy is a use case's (ADR-0041) |
| Leave `sha256` `None` for a model this project produced | The one case where the file is right there and just written. ADR-0040's rule already says who computes a checksum |
| Fix the cancel defect in a later task | It is the run this task's record cannot describe, and the fix is four lines and one contract assertion |

## Compliance

- `tests/integration/test_training_history.py` — 20 tests: a run, its config, its device, its every
  epoch and its error round-trip; **the project is closed and reopened**; a run with nothing held
  out comes back with no `validation` block on any epoch; a second save advances one row rather than
  making two; a succeeded run registers a model with the class map, the input size, the checksum and
  the provenance; a cancelled and a failed run register nothing and are still recorded.
- `tests/unit/test_training_cancellation.py` — the measured defect, deterministically and without
  ultralytics: one worker, occupied, so the job is queued and dropped. It fails on the code before
  the fix.
- `tests/contract/training_provider.py` — the fifteenth assertion, satisfied by the fake unchanged
  and by `LocalTrainingProvider` in the `slow` gate.
- `tests/integration/test_model_registry.py` — the checksum is computed from the file, a given one
  is kept, and an absent file stays `None`.
- `tests/integration/schema_history.py` — v9's tables are in the revert map, which its own guard
  test enforces.
- The golden is byte-identical. Nothing in this task is on a numerical path.

## References

- ADR-0080 / M8-T01 — the port, the vocabulary, the snapshot, and *a run has to be findable after a restart*
- ADR-0082 / M8-T03 — the provider this records, and the validation block that means a held-out set existed
- ADR-0006 — the seam, and the compliance clause §5 closes
- ADR-0043 — the job runner, cancellation, and the dropped job §7 fixes
- ADR-0031 / ADR-0042 / ADR-0044 — a block present in full or absent in full; file or table, and the test that decides
- ADR-0050 / ADR-0040 — the model registry, and who computes a checksum
- ADR-0041 — *a use case earns its place*; §4 and the unwired composition root
