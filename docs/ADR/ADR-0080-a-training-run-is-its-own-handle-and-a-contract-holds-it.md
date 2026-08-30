# ADR-0080 — A training run is its own handle, and a contract test holds the port

- **Status:** Accepted
- **Date:** 2026-08-30
- **Deciders:** operator + agent (M8-T01)
- **Affects:** `core/entities/training.py`, `core/ports/training.py`, `tests/contract/` · M8

## Context

ADR-0006 chose this seam in M0, before any of it existed, and gave the reasons: training runs
for hours instead of seconds, consumes a dataset rather than an image, produces artifacts and
metrics rather than detections, must be cancellable, and may run on another machine. *"The
tempting shortcut is to add a `train()` method next to `detect()` on the detector. That is the
wrong seam."*

Four milestones on, three facts constrain how that port is written.

**M2-T08 defined seven ports and shipped one.** The other six had no implementation, no caller
and no second candidate, and `core/ports/__init__.py` wrote the rule into its own docstring:
*the rest ship with their first adapter*. ADR-0041 sharpened it into the project's most-applied
rule — *a use case earns its place or is not written* — and it has been applied four times since.
Writing `TrainingProvider` now, with no adapter, is that rule's clearest violation.

**ADR-0043 already settled the thread policy.** `JobRunner.submit` gives progress, cooperative
cancellation and a listener that fires on the worker thread. A second policy for training would
be the thing that ADR governs, duplicated.

**ADR-0031 already met "several producers, several schemas"** — four measurement producers with
four column sets — and answered with a core plus blocks that are present in full or absent in
full. Trainers report different numbers for the same reason.

M7-T09 built the input side: annotations leave as labels a trainer reads. Nothing in this project
produces a model.

## Decision

### 1. The port is written first, and a contract suite is what makes that legitimate

`core/ports/training.py` lands in M8-T01 with no adapter, which the package's own docstring
forbids. The exception is written into that docstring rather than left as a silent departure.

The objection to an unimplemented port is not that it is empty — it is that it is
**unfalsifiable**. Nothing can disagree with it, so it survives until real code has to fit
through it, by which time it is quoted in a document and looks decided. That is exactly what
happened to M2-T08's six.

So the deliverable is not the `Protocol`. It is `tests/contract/training_provider.py`: fourteen
assertions that a `FakeTrainingProvider` satisfies today and `LocalTrainingProvider` must satisfy
in M8-T03, unchanged. This is ADR-0006's own compliance clause — *both providers pass the same
contract test suite* — written as a test rather than as a review note, and it is what lets the
port be wrong now, cheaply, instead of in M8-T07 when a second implementation discovers it.

The suite polls `status()` rather than waiting on the listener, deliberately: that is what a
provider on the other side of a network has to do anyway, so a provider that worked only through
its callback would pass a test it should not.

### 2. A run is not a `Job`, and the five duplicated state names are the decision

`TrainingStatus` has the same five values as `JobState`, in a different enum, in a different
layer. That is duplication and it is deliberate:

- a `Job` is **in-process and dies with the process**; a training run has to be findable after a
  restart (M8-T04);
- a `Job` wraps a callable this application submitted; a run may be executing on a machine this
  application did not start (M8-T07);
- `core` may not import `application` at all (PROJECT_RULES §2.1), and `jobs.py` is in
  `application`. Moving the job runner into `core` to share an enum would put a thread pool in
  the pure layer to save five strings.

What must **not** happen is a second thread policy. `LocalTrainingProvider` drives its run with
the `JobRunner` underneath — ADR-0043's checkpoints, ADR-0043's listener, ADR-0043's honesty
about what cancel can promise — and a remote one polls. The port describes the *run*; how it is
executed stays where ADR-0043 put it.

### 3. `TrainingRun` is the handle, and every observation is a snapshot

`start` returns a `TrainingRun`; `status(run_id)` returns a fresh one; `cancel(run_id)` takes the
same id. There is no separate `TrainingHandle` class.

A handle that is a live view of a running object cannot describe a run on another machine, and
cannot be stored. A frozen snapshot can be both — which is what lets M8-T04 persist the record it
was handed, and M8-T05 draw a chart from a list of them without holding anything alive.

### 4. A metric is an epoch and named scalars, and the vocabulary is declared once

```
loss        train_loss                                          always
validation  val_loss, precision, recall, map50, map50_95        a held-out set existed
```

ADR-0031's rule, second application: **one quantity, one name**, and a **block present in full or
absent in full**. The split is the one the work has — a trainer always has a training loss, and
everything else exists only if a validation pass ran, which is exactly what
`DatasetSpec.val_images == 0` describes.

Not a wide record with `NaN` where a run did not validate. A dataset with nothing held out has no
precision; it did not measure one and lose it. Six ADRs in M3 turned on the difference between
absent and substituted, and `EpochMetrics.__post_init__` refuses an unknown name and a half-filled
block — in the constructor, not in a validator a caller must remember, because the caller is a
framework callback firing once a minute for six hours and the first wrong name should fail on
epoch 1.

### 5. There is no `collect_artifacts()`; the configuration says where the weights go

`TrainingConfig.output_directory` is project-relative, under `models/` (PROJECT_RULES §5), and a
succeeded run carries `weights_path` pointing at a file that exists.

A fourth method would make "the run succeeded" and "the file is here" two facts that can disagree,
and the disagreement surfaces in M8-T04 when registering a model fails for a run reported green.
Making it one fact puts the file move where the knowledge is — a copy locally, a download
remotely — and turns ADR-0006's *no silent artifacts on disk* into an assertion the contract suite
makes on both providers.

### 6. Cancellation means what ADR-0043 made it mean, at an epoch boundary

*Stop at the next checkpoint*, and a trainer's checkpoint is the end of an epoch. A run cancelled
two minutes into a forty-minute epoch keeps training for thirty-eight. **The UI says so** (M8-T05),
because M5-T07 already learned that a button which appears to do nothing produces an operator who
concludes the application has hung.

A cancelled run keeps the epochs it completed, the way ADR-0043's cancelled import keeps the files
it already copied. Cancelling an unknown or finished run does not raise: the caller is a button
that can be pressed twice.

### 7. `core` names no framework

`DatasetSpec` carries a directory, class names and two counts. What file lives inside that
directory is the builder's decision (M8-T02) and the provider's to read. PROJECT_RULES §2.5 keeps
`"yolo"` out of `gui/` and `core/science`; naming a framework's manifest file in `core/entities`
would decide for the second provider what the first happened to use.

## Consequences

**Positive**

- M8-T02…T05 have a vocabulary to write against, and M8-T03 inherits fourteen tests instead of
  writing its own.
- The port can be found wrong now. A contract suite is the only form of "we will keep this
  honest" that has ever survived in this repository.
- One thread policy. ADR-0043 keeps the only checkpoint-and-listener design in the project.
- A run record is storable and transmissible by construction, so M8-T04 and M8-T07 do not have to
  re-shape it.
- The metric vocabulary exists before the first producer, which is the ordering ADR-0031 wished it
  had had.

**Negative**

- **A port with no adapter, which this package's docstring forbids.** The contract suite is an
  argument, not a proof: M8-T03 may still find the shape wrong, and then this ADR is superseded
  rather than vindicated. That is the cost, and it is named here rather than discovered later.
- The fake provider is code that ships nothing to an operator. It is the price of §1.
- Five state names exist twice. A reader who finds `TrainingStatus.RUNNING` and `JobState.RUNNING`
  has to be told why, which is what §2 is.
- `METRIC_BLOCKS` will need a new block the first time a trainer reports something real that is
  not in it — a learning rate, a per-class mAP. Adding one is a one-line change plus an ADR-0031
  style argument; guessing them now is the thing ADR-0031 says not to do.
- No `resume`. A multi-hour run interrupted by a crash starts again. ADR-0006 lists resumption as
  a requirement and this port does not serve it — named, not missed; it needs a stored checkpoint
  path, which is M8-T04's record and not this task's port.

**Neutral**

- The contract suite polls, which costs a poll interval of latency in tests and nothing in
  production.
- `TrainingRun` is frozen, so a provider replaces the snapshot rather than mutating it. That is
  more allocation and less to reason about.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Write the port in M8-T03, from what ultralytics needed | Then the port *is* ultralytics' shape with `abstract` on it, and M8-T07 pays. ADR-0006 committed to two implementations in M0 for precisely this reason |
| `train()` on `Detector` | ADR-0006 refused it: a multi-hour, dataset-consuming, artifact-producing call inside the object used for per-image inference, and LoG would have to answer for a method it has no meaning for |
| A training run **is** a `Job` | `core` cannot import `application`; a `Job` dies with the process; a remote run was never submitted to a thread pool here |
| A separate `TrainingHandle` with live state | Cannot describe a remote run, cannot be persisted. The snapshot is both |
| Metrics as a free `dict[str, float]` | ADR-0031's exact starting position: two providers, two spellings of one quantity, and a chart that cannot tell they are the same series |
| Metrics as a fixed wide record with `NaN` | Says a run with no held-out set *has* a precision and it is missing. It has none — ADR-0031, and ADR-0025 before it |
| `collect_artifacts(run_id, into)` as a fourth method | Makes "succeeded" and "the file is here" two facts that can disagree, and the disagreement lands on M8-T04 |
| Progress as a `Progress(done, total)` mirrored from `jobs.py` | `epochs_done` is derived from the last metric, so the bar and the chart cannot disagree; and `total` is already `config.epochs` |
| No listener — the UI polls `status()` | Would work, and would leave M5's marshalling adapter (ADR-0058) with nothing to marshal for the one long job this milestone adds. The pattern is already in the codebase; a second one is not needed |

## Compliance

- `tests/contract/training_provider.py` is the suite; `tests/contract/test_fake_training_provider.py`
  runs it against `FakeTrainingProvider`, twice — with a held-out set and without — and M8-T03 adds
  one file beside it with three fixtures and no new assertions.
- One test in that file proves **the suite can fail**: a provider that reports success and leaves
  no weights behind is caught, which is ADR-0006's *no silent artifacts on disk* as an assertion.
- `tests/unit/test_training_entities.py` pins the refusals: an unknown metric name, a half-filled
  block, a dataset with no training image, a configuration of zero epochs.
- `tests/unit/test_import_graph.py::test_training_and_inference_stay_apart` enforces ADR-0006's
  structural clause — nothing under `infrastructure/models/` imports `infrastructure/training/`,
  and the reverse the day that directory exists.
- `nanoscope.core.ports` is still importable without torch, ultralytics or Qt; the existing weight
  check covers it.
- No number moved: the characterization golden is byte-identical, which it must be — nothing in
  this task computes one.

## References

- ADR-0006 — the seam, and the compliance clause this task turns into tests
- ADR-0041 — *a use case earns its place*; §1 is the argument for the exception
- ADR-0043 — cancellation, the listener, and the thread policy §2 declines to duplicate
- ADR-0031 — a core plus blocks, present in full or absent in full
- ADR-0050 — the registry that receives what a run produces
- ADR-0078 / M7-T09 — the labels a dataset is built from
- `docs/Architecture.md` §4.3, `docs/CURRENT_TASK.md` (M8-T01)
