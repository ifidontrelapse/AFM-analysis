# ADR-0082 — A validation block means a held-out set existed, not that a validation pass ran

- **Status:** Accepted
- **Date:** 2026-08-30
- **Deciders:** operator + agent (M8-T03)
- **Affects:** `infrastructure/training/local.py`, `application/use_cases/dataset.py` · M8

## Context

M8-T01 wrote `TrainingProvider` before any adapter and justified that with fourteen assertions a
second implementation would have to satisfy unchanged (ADR-0080 §1). M8-T02 built the dataset.
This task is `LocalTrainingProvider`, and the thing that judges it is the existing suite passing
with three new fixtures and no new assertions.

Facts measured against ultralytics 8.4.41 before anything was written, rather than assumed:

| What | Measured |
|---|---|
| Where an epoch ends | `on_fit_epoch_end`, with `if self.stop: break` immediately after it |
| Whether `trainer.stop = True` works | Asked for 8 epochs, stopped after 2, `best.pt` still on disk |
| How often that callback fires | **Four times for three epochs** — `[0, 1, 2, 2]` |
| The metric names | `metrics/precision(B)`, `metrics/mAP50-95(B)`, `val/box_loss`, `train/box_loss`, … |
| Epoch numbering | 0-based; `EpochMetrics.epoch` is 1-based |
| Cost of a contract-sized run | **2.7 s** — 3 epochs, 2 training images, 1 held out, 32 px, CPU |

And one found by the contract suite rather than by reading: **a dataset M8-T02 builds with
`val_fraction=0.0` cannot be trained at all.** Ultralytics refuses a manifest whose `val` does not
resolve, and M8-T02 never created `images/val` for a split that held nothing out. It failed with
*"Dataset error"* before the first epoch.

## Decision

### 1. `val` in `data.yaml` always resolves, and points at the training split when nothing is held out

Because the trainer requires it to, and because it **validates the final epoch whether or not it was
asked to** — `if self.args.val or final_epoch: self.metrics = self.validate()`. The directory has to
exist; `val=False` is not enough.

The manifest says so in a comment, in the file, where a person reading the dataset will see it.

### 2. The provider does not report those numbers as validation

This is the decision the title names, and the contract test that failed is what forced it.

A precision computed on the training set is **the model scored on what it trained on** — the
self-confirmation ADR-0044 named one level down, dressed as a metric. ADR-0080's `validation` block
means *a held-out set existed*, and `LocalTrainingProvider` knows `dataset.val_images == 0`, so it
omits the block entirely — including for the final epoch, where ultralytics computes one anyway.

Without this, a run with no held-out set reports `train_loss` for five epochs and then a sixth epoch
carrying a precision, a recall and two mAPs out of nowhere. Every one of those numbers would be
wrong in the direction of looking better, and a chart would show them as the run's result.

### 3. Every epoch once, in order, which means deduplicating

`on_fit_epoch_end` fires again for the last epoch after the final validation. The port promises one
entry per epoch, never sparse, so the reporter keeps the last number it emitted and drops a repeat.
An off-by-one in every chart, avoided by measuring the callback rather than assuming it.

### 4. Cancellation sets the trainer's flag; it does not raise

`JobRunner` owns the flag (ADR-0080 §2 — no second thread policy), and the epoch boundary is where it
is read. `trainer.stop = True`, not `raise`: raising out of a framework callback abandons the
checkpoint that ADR-0080 promised a cancelled run would keep, and ADR-0043's cancelled import keeps
the files it already copied for the same reason.

### 5. The metric names are translated in the adapter

`metrics/mAP50-95(B)` is ultralytics' name for `map50_95`. The port declared the vocabulary once so
two providers could not spell one quantity two ways (ADR-0031's rule); translating is therefore the
adapter's job, not the reader's. `train_loss` and `val_loss` are the **sums** of the three components
ultralytics reports — a total is what a chart plots, and the split into box/cls/dfl is a framework's
internal rather than a quantity this project has named.

### 6. The contract subclass is `slow` and stays in the gate

2.7 s a run, tens of seconds for the suite. An environment variable guarding it would make it a test
nobody runs, and a test nobody runs is one that rots. PROJECT_RULES §6 keeps *inference* out of the
gate because it is not reproducible, not because it is slow — and nothing here asserts a number a
model produced. It skips where ultralytics is absent, which is CI by design (M1-T08), and the skip
names what is missing.

Trained from `yolo11n.yaml`, which ships inside ultralytics, so the run needs no download and no
checkpoint. The model it produces is useless, which is the point.

### 7. Nothing is wired into the composition root

ADR-0041's rule, sixth application. The provider needs the job runner and the device manager, both of
which the container already holds; the wiring is five lines and arrives with M8-T05's window.

## Consequences

**Positive**

- **ADR-0080 §1 is settled rather than argued.** The suite written against a fake passed against a
  real trainer with three fixtures and no edits, and it found a defect on the way.
- A training run shows progress and cancels through the same `Job` as an import, so M5-T07's status
  widget will display one without knowing what training is.
- A run with no held-out set reports no validation numbers, instead of reporting flattering ones.
- The ADR-0006 separation guard is non-vacuous in both directions for the first time.

**Negative**

- **The gate grew by about a minute** on a machine with torch installed. Named, and judged worth it
  against the alternative of an opt-in test.
- Ultralytics' callback names, metric keys and `stop` attribute are **not a public API**. A version
  bump can move all three, and the failure would be a run that reports nothing rather than a crash.
  What protects it is the contract suite running on every gate, which is why it is not opt-in.
- The manifest points `val` at the training split when nothing is held out. Anyone reading
  `data.yaml` without reading the comment could believe there is a validation set. Mitigated in the
  file, not in code.
- Training from `yolo11n.yaml` exercises the port and nothing about model quality. That is M8-T08's
  job and it needs real weights and real scans.

**Neutral**

- Ultralytics writes `last.pt` beside `best.pt`; only `best.pt` is reported. `TrainingRun` has one
  `weights_path`, and the second file is on disk for anyone who wants it.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Forward `trainer.metrics` unchanged | Two providers, two spellings of one quantity — ADR-0031's exact starting position |
| Report the final-epoch validation even with nothing held out | The model scored on its own training data, reported as a validation metric. Wrong in the direction of looking better |
| Omit `val` from `data.yaml` when nothing is held out | Ultralytics refuses the manifest. Measured, not assumed |
| Raise out of the callback to cancel | Abandons the checkpoint ADR-0080 promised a cancelled run keeps |
| A second cancellation flag on the provider | ADR-0080 §2 forbids exactly this: the runner's flag or nothing |
| Report `box_loss`, `cls_loss`, `dfl_loss` separately | Three names this project has not defined, in a vocabulary declared once on purpose |
| Guard the contract subclass behind an environment variable | A test nobody runs by default is a test that rots, and this one costs seconds |
| Train from `checkpoints/best12x.pt` in the contract | Weights are not committed (PROJECT_RULES §7), so the test would pass on one machine |
| Wire it into the container now | No caller. ADR-0041, sixth application |

## Compliance

- `tests/contract/test_local_training_provider.py` — `TrainingProviderContract` with three fixtures
  and **no new assertions**, twice: with a held-out set and without. 26 tests.
- The no-validation subclass inherits the assertion that made this ADR necessary: every epoch's
  `validation` block is absent, including the last.
- `tests/integration/test_dataset_builder.py` pins the manifest fallback and the comment in it.
- `tests/unit/test_import_graph.py::test_training_and_inference_stay_apart` now covers both
  directions of ADR-0006's clause — six cases where there were four, and the two new ones are
  `infrastructure/training/`.
- The golden is byte-identical. Nothing in this task is on a numerical path.

## References

- ADR-0080 / M8-T01 — the port, the vocabulary, and the contract this settles
- ADR-0081 / M8-T02 — the dataset, and the manifest this task had to change
- ADR-0043 — the job runner, the checkpoint, and what cancel may promise
- ADR-0031 — one quantity, one name; a block present in full or absent in full
- ADR-0044 — a model trained on its own output is confirming itself
- ADR-0004, ADR-0049 — the device, and why a fallback is reported
- PROJECT_RULES §6 — what is in the gate, and why
