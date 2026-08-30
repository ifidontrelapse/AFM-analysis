# CURRENT TASK

**ID:** `M8-T02`
**Title:** The dataset builder: annotations become something a trainer can read
**Milestone:** M8 — Training module, second task
**Defect:** — · **ADR:** **ADR-0081** (to be written)
**Branch:** `feat/m8-training`
**Status:** **planning 2026-08-30.** Not started.

---

## Why this task is second

M8-T01 declared what a training run consumes: a `DatasetSpec` — a directory, class names, and how
many images are in each half. Nothing produces one. M8-T03 trains from one and cannot be written
until something does.

M7-T09 built half of it and **said which half it was not building**: it writes
`labels/<stem>.txt` in the trainer's own format and a `classes.txt` whose line numbers are the
indices, and ADR-0078 stopped there on purpose — *"a split is a dataset decision — how much to hold
out, stratified by what — and it belongs to M8-T02 rather than to the task that happened to write
the labels first."* This is that task.

---

## The decision this task actually turns on

**A height map is not an image, and a trainer only reads images.**

The scans in a project are `.spm` and `scan.000`: 2-D `float32` arrays in nanometres, sometimes
negative, with a range that depends on the sample. Ultralytics reads PNG and JPEG. So the builder
has to *make a picture*, and every choice in doing so decides what the model learns.

There is exactly one right answer available and it is already written down: **prepare a training
image the way `YoloDetector._prepare_image` prepares an inference input.** A model trained on one
distribution and used on another is a model measured on a question nobody asked, and three ADRs
already settled what that preparation is:

- **ADR-0015** — normalise in floating point, *then* cast to `uint8`. Casting first keeps only the
  integers inside the map's range and wraps the rest; on `afm_sparse_low_snr` the result was
  **anti-correlated (r = −0.499)** with the correct image.
- **ADR-0023** — invert for `BRIGHT_ON_DARK`, which is AFM's convention, and not for TEM where the
  particles are already dark.
- **ADR-0016** — letterbox isotropically. **Which this builder must *not* do**, because ultralytics
  letterboxes to `imgsz` itself at train time and transforms the labels with it. Doing it here
  would letterbox twice and squeeze every particle into the middle of a doubly-padded frame.

So: **native resolution, min-max to `uint8`, invert by polarity, PNG.** The geometry stays the
trainer's.

**And the array is `z_above`, not the file.** `detect()` is handed `z_flat - substrate` — that is
what `PipelineResult` computes and what every detection in this project was ever made from. A
dataset built from raw height maps would train a model on tilt and substrate that inference has
already removed. It costs a `run_preprocessing` per image, which is the expensive path and the
correct one.

---

## The other decisions

**1. The split is by image, never by box.**

Two boxes from one scan, one in train and one in val, is leakage: the val score then measures how
well the model memorised that scan's substrate, instrument noise and particle population. Every box
of a scan goes to the same side. This is the one thing in the task that a reviewer cannot see from
the output and that quietly inflates every number M8-T08 will report.

**2. The split is deterministic, and the seed is recorded.**

A rebuild that shuffles differently makes two runs incomparable. Seeded, and the seed goes in
`data.yaml` where a person can read it.

**3. Where it goes: `cache/`.**

PROJECT_RULES §5 fixes the layout and says *anything under `cache/` must be safely deletable at any
time without data loss*. A built dataset is derived from annotations that are still in the database
— it is re-creatable by definition, which is exactly what `cache/` means. `exports/` is what an
operator takes away (ADR-0067); a dataset is what the application feeds itself.

The consequence, stated rather than discovered: **deleting `cache/` after a run leaves the run's
`DatasetSpec.root` pointing at nothing.** That is why `DatasetSpec` carries `classes` and both
counts — M8-T01 wrote *"a run has to be readable a month later, when the directory may be gone"* —
and M8-T04 persists them.

**4. Which annotations, and the caller says.**

ADR-0044's `source` is load-bearing: *a model trained on its own output is confirming itself.*
M7-T09 made the caller name the scope rather than defaulting to one that hides the question, and
this builder takes the same argument for the same reason.

**5. `_to_label` is reused, not copied.**

The one thing today already cost: `display.py` kept a second copy of a four-entry extension map and
an operator's folder of scans would not open. The normalisation, the clamping and the six decimal
places live in `use_cases/annotations.py`; this builder imports them.

---

## Scope

**In scope**

1. `application/use_cases/dataset.py` — `build_dataset(repository, *, sources, val_fraction, seed)`
   returning `DatasetSpec`
2. The image preparation shared with inference, in **one** place both call
3. `data.yaml`, the split, and the `images/{train,val}` + `labels/{train,val}` layout
4. **ADR-0081**
5. Tests: the split is by image, deterministic, and the prepared image matches inference's

**Out of scope**

- **Training anything** — M8-T03, and it brings ultralytics with it
- **Any UI** — M8-T05. The builder is a use case with no dialog in front of it yet
- **Persisting the dataset in the database** — M8-T04 persists the *run*, which carries the spec
- **Augmentation** — ultralytics does its own, and a second source of it is two policies

---

## Definition of done

- [ ] `build_dataset` returns a `DatasetSpec` that `FakeTrainingProvider` accepts
- [ ] One image preparation, called by both the detector and the builder, and a test that says so
- [ ] The split is by image and deterministic, both asserted
- [ ] ADR-0081 + the ADR index
- [ ] `make check` green, golden byte-identical — moving `_prepare_image` must move no number
- [ ] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [ ] Commit: `M8-T02: annotations become a dataset, prepared the way inference prepares`
