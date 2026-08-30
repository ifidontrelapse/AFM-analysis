# CURRENT TASK

**ID:** `M8-T03`
**Title:** `LocalTrainingProvider` — the first thing in this project that produces a model
**Milestone:** M8 — Training module, third task
**Defect:** — · **ADR:** **ADR-0082** (to be written)
**Branch:** `feat/m8-training`
**Status:** **planning 2026-08-30.** Not started.

---

## Why this task is third

M8-T01 wrote the port and, with it, **fourteen assertions a second implementation must satisfy** —
`tests/contract/training_provider.py`, run against a fake. M8-T02 built the dataset the port
consumes. This is the task those two were written for, and the deliverable that judges it is not new
tests: it is **the existing suite passing with three new fixtures and no new assertions.** If it
does not, ADR-0080 §1 was wrong and says so itself.

ADR-0006 chose this seam in M0: *"`LocalTrainingProvider` — trains on this machine (ultralytics),
device resolved by the Device Manager."*

---

## What was measured before planning

Not assumed — run, on this machine, against ultralytics 8.4.41:

**1. `on_fit_epoch_end` is the epoch boundary, and `trainer.stop = True` inside it stops the run.**
The trainer's loop reads `if self.stop: break` immediately after firing that callback. Asked for 8
epochs, stopped after 2, and `best.pt` was still on disk — which is exactly ADR-0043's *stop at the
next checkpoint* and ADR-0080's *a cancelled run keeps the epochs it completed*.

**2. The callback fires twice for the last epoch.** Three epochs reported `[0, 1, 2, 2]`: the final
`val` after the loop fires it again. **The port promises one entry per epoch, in order, never
sparse** (M8-T01's contract asserts it), so the adapter deduplicates by epoch number. A trap that
would have shipped as an off-by-one in every chart.

**3. The metric names, read off a real run:**

```
trainer.metrics            metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)
                           metrics/mAP50-95(B)   val/box_loss  val/cls_loss  val/dfl_loss
trainer.label_loss_items() train/box_loss  train/cls_loss  train/dfl_loss
```

Ultralytics epochs are **0-based**; `EpochMetrics.epoch` is 1-based (M8-T01), so `+ 1`.

**4. A model builds from a YAML that ships with the package** — `yolo11n.yaml` resolves inside
ultralytics' own `cfg/models/11/`, so a contract run needs no download and no checkpoint.

**5. It costs 2.7 s.** Three epochs, two training images, one held out, 32 px, CPU. So the contract
subclass is **`slow` and in the gate**, not hidden behind an environment variable — a test nobody
runs by default is a test that rots, and this one is affordable.

---

## The decisions

**1. The job runner underneath, exactly as ADR-0080 §2 said.**

*"The local provider drives it with the `JobRunner` underneath; a remote one polls. What must not
happen is a second thread policy in the layer ADR-0043 already settled."* So `start` submits to the
runner, the run's body reports progress through `JobContext.report(epoch, epochs)` — which makes
M5-T07's job status widget work for training with no new code — and cancellation is the runner's
flag, **read at the trainer's epoch boundary** rather than raised. Raised would abandon the
checkpoint; ADR-0080 promised it is kept.

**2. Metrics are mapped, not forwarded.**

`metrics/mAP50-95(B)` is ultralytics' name for a quantity ADR-0080 calls `map50_95`. The port
declared the vocabulary once so two providers cannot spell one quantity two ways (ADR-0031's rule),
and the adapter is where the translation belongs. `train_loss` and `val_loss` are the **sums** of the
three components ultralytics reports — a total is what a chart plots, and the split is a
framework's internal, not a quantity this project has named.

The `validation` block is present only when it is all there, which is what `val_images == 0` means:
the run is started with `val=False` and none of the five keys exist.

**3. The device is the manager's answer, and the run records what it got.**

`DeviceProvider.select(config.device)` (ADR-0004, PROJECT_RULES §2.6 — nothing else asks torch), and
`Device.torch_name` is what ultralytics is handed. `TrainingRun.device` carries the resolved one, not
the requested one, because a run that fell back to CPU took forty times as long for a reason that
must be in the record (ADR-0049).

**4. Artifacts land where the configuration said, and the path is read back.**

`TrainingConfig.output_directory` under the project root, ultralytics' `project` / `name` split
across it. The path on the run is `trainer.best` **relative to the project root**, read back rather
than assembled: ultralytics increments the directory name on collision, and a path this adapter
composed would name a directory the trainer did not use.

**5. Nothing is wired into the container.**

ADR-0041's rule, sixth application: the composition root gains a `TrainingProvider` when M8-T05 has
a window that asks for one. It needs the runner and the device manager, both of which the container
already holds, so the wiring is five lines whenever it is earned.

---

## Scope

**In scope**

1. `infrastructure/training/local.py` — `LocalTrainingProvider`
2. `tests/contract/test_local_training_provider.py` — three fixtures, no new assertions
3. **ADR-0082**
4. The import guard from M8-T01 stops being half-vacuous: `infrastructure/training/` now exists

**Out of scope**

- **Registering the produced model** — M8-T04 persists the run and registers the `ModelDescriptor`
- **Any UI** — M8-T05
- **Resumption** — named in ADR-0080's negatives; it needs a stored checkpoint path, which is
  M8-T04's record
- **The remote protocol** — M8-T07

---

## Definition of done

- [ ] `LocalTrainingProvider` passes `TrainingProviderContract` unchanged
- [ ] Every epoch reported once, in order, with the vocabulary the port declared
- [ ] Cancel stops at an epoch boundary and keeps what was trained
- [ ] ADR-0082 + the ADR index
- [ ] `make check` green, golden byte-identical
- [ ] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [ ] Commit: `M8-T03: the first thing in this project that produces a model`
