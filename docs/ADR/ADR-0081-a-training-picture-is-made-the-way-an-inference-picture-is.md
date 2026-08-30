# ADR-0081 — A training picture is made the way an inference picture is

- **Status:** Accepted
- **Date:** 2026-08-30
- **Deciders:** operator + agent (M8-T02)
- **Affects:** `infrastructure/imaging/network_input.py`, `infrastructure/models/yolo.py`,
  `application/use_cases/dataset.py`, `core/ports/project_repository.py` · M8

## Context

M8-T01 declared what a training run consumes — a `DatasetSpec`: a directory, class names, and how
many images are in each half. Nothing produced one, and M8-T03 cannot be written until something
does.

M7-T09 built half of it and named the half it was not building. ADR-0078 stopped before `data.yaml`
and the split on purpose: *"a split is a dataset decision — how much to hold out, stratified by what
— and it belongs to M8-T02 rather than to the task that happened to write the labels first."*

The fact that governs everything else: **the scans in a project are not images.** They are 2-D
`float32` arrays in nanometres, often negative, with a range that depends on the sample —
`afm_sparse_low_snr` spans 9.3 nm and `tem_dark_particles` spans 210. Ultralytics reads PNG and
JPEG. Something has to make a picture.

Three ADRs already decided how, for the other direction:

- **ADR-0015** — normalise in floating point, then cast. Casting first keeps only the integers
  inside the map's range and wraps the rest; on one phantom the result was **anti-correlated
  (r = −0.499)** with the correct image. Rated critical, shipped alone.
- **ADR-0023** — invert for `BRIGHT_ON_DARK`, not for TEM where particles are already dark.
- **ADR-0016** — letterbox isotropically, and pad with 255 so the border reads as more substrate.

## Decision

### 1. One function makes the picture, and both sides call it

`infrastructure/imaging/network_input.py::as_network_input(z, *, polarity)` — min-max to `uint8`,
inverted by polarity, **same shape in and out**. `YoloDetector._prepare_image` calls it and so does
the dataset builder.

A model trained on pictures made one way and used on pictures made another is measured on a question
nobody asked, and **the failure is silent**: no exception, no wrong shape, just a detector that is
worse than it should be for a reason nobody can see. The copy is what makes it happen, and this
repository learned that on 2026-08-30 — `display.py` kept a second copy of a four-entry extension map
and an operator's folder of scans would not open. That one produced an error message. This one would
not.

`imaging` rather than `models`: it belongs to neither side of ADR-0006's inference/training
separation, and both sides import it.

### 2. Letterboxing stays out of it

`as_network_input` does no resizing and no padding, because the two callers need different geometry:
the detector pads to its own square input and inverts that map in `_scale_boxes`, while **ultralytics
letterboxes to `imgsz` itself at train time and transforms the labels with it.** Doing it in the
builder too would letterbox twice and squeeze every particle into the middle of a doubly-padded
frame.

The consequence, stated rather than discovered: the detector normalises *after* downscaling and the
builder normalises at full resolution, so the two windows are not bit-identical on the same scan.
That is the right way round — the trainer's own resize happens on the `uint8` picture, after the
stretch, which is what inference does too.

### 3. The array is `z_above`, not the file

`detect` is handed `z_flat - substrate`. A dataset built from raw height maps would teach a model
the tilt and the substrate that inference has already removed. It costs a `run_preprocessing` per
scan, which is the expensive path and the correct one. SEM and TEM have no substrate to build
(ADR-0031) and are analysed as they are, so the loaded image is the picture.

### 4. The split is by image, never by box

Two boxes off one scan, one in each half, is **leakage**: the validation score then measures how
well the model memorised that scan's substrate, its instrument noise and its particle population,
and every number M8-T08 reports is quietly inflated by it. This is the one decision in the task a
reviewer cannot see from the output, which is why it has a test of its own.

Seeded, and the seed is written into `data.yaml`: two runs that split differently cannot be
compared. Rounded **down and then not up** — asking for a fifth of four scans is asking for 0.8, and
holding out one of four is a 25% validation set reported as 20%. Zero is the honest answer, and
`val_images == 0` already means something specific (ADR-0080: the `validation` metric block is
absent, not `NaN`).

### 5. It lands in `cache/`

PROJECT_RULES §5: *anything under `cache/` must be safely deletable at any time without data loss.*
A built dataset is derived from annotations that are still in the database — re-creatable by
definition, which is what `cache/` means. `exports/` is what an operator takes away (ADR-0067); this
is what the application feeds itself.

The consequence, stated: **deleting `cache/` after a run leaves that run's `DatasetSpec.root`
pointing at nothing.** That is exactly why M8-T01 put `classes` and both counts *on the spec* — *"a
run has to be readable a month later, when the directory may be gone"* — and why M8-T04 persists
them rather than a path.

Two port methods arrive with it, `write_cache_text` and `write_cache_image`, because `application`
may not touch the filesystem and encoding a PNG is `cv2` (Architecture §3.2, the division ADR-0073
already made for a painted mask). Neither is `@_serialised`: they write files and touch no row, and
holding the repository lock across a directory of a hundred scans would make this the one thing in
the project that stops every other job.

### 6. `_to_label` is reused, not copied

The normalisation, the clamping and the six decimal places live in `use_cases/annotations.py`, and
this builder imports them. Same rule as §1, one layer up.

### 7. A scan that cannot be prepared is reported, not fatal

Eleven usable scans and one unreadable one is a dataset. The names and reasons come back on the
report (ADR-0040's rule, from the building side); only *no* usable scan is a refusal, because an
empty dataset is indistinguishable from "nothing was drawn" (ADR-0048's rule, third site).

## Consequences

**Positive**

- Training and inference cannot drift apart without a test failing, and the test names the function
  identity rather than comparing outputs of two copies.
- M8-T03 has a dataset to train from and does not have to decide any of this.
- The leakage question is answered once, before any number is reported, instead of being discovered
  when M8-T08's scores look too good.
- `data.yaml` carries the seed, so two runs are comparable or visibly are not.

**Negative**

- **Building a dataset preprocesses every scan**, which is the slow path — seconds per scan, minutes
  for a project. It is not a job yet (no UI asks for one until M8-T05), so a large project blocks
  its caller. Named here rather than solved: the runner exists (ADR-0043) and the caller that needs
  it does not.
- `data.yaml` is written by hand rather than with a YAML library. Four keys and a list of short
  names did not justify a dependency; the quoting is the one thing that could go wrong, so every
  name is quoted and an embedded quote is doubled.
- The two callers of `as_network_input` normalise over different windows (§2). Deliberate, and the
  alternative — normalising before the detector's resize — would move every number in the golden.
- A dataset under `cache/` can be deleted out from under a finished run's `root`. §5's consequence,
  and the reason the spec carries what it carries.

**Neutral**

- No augmentation. Ultralytics does its own, and a second source of it is two policies.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Copy the four lines of normalise-and-invert into the builder | The exact shape of the bug found on 2026-08-30, with a worse failure mode: a silently worse model instead of an error message |
| Render with the viewer's colormap (`afm_to_rgb`) | That is a *display* transform — percentile-clipped and coloured for a person. Inference has never seen it |
| Write pre-letterboxed 640×640 images | Ultralytics letterboxes to `imgsz` itself and transforms the labels; the result is padding inside padding |
| Build from the raw height map | Trains the model on tilt and substrate that `detect` never sees |
| Split by box | Leakage. The validation score stops measuring generalisation and starts measuring memory |
| Round the validation count up | Holding out one of four scans is 25% reported as 20%. A number that is wrong in the direction of looking better |
| Put the dataset in `exports/` | An export is what an operator takes away (ADR-0067). This is an intermediate the application feeds itself, and it is re-creatable |
| A YAML library for `data.yaml` | A dependency for four keys, in a layer that has none of its own |
| Let the builder write files itself | `application` may not touch the filesystem (Architecture §3.2), and encoding a PNG is `cv2` |

## Compliance

- `tests/integration/test_dataset_builder.py` — nineteen tests. The two that carry the decision:
  **the written PNG equals `as_network_input(preprocess_image(...).z_result, …)` byte for byte**, and
  **no scan's stem appears in both halves**.
- One test asserts `yolo.as_network_input is as_network_input` — the *same function*, not two that
  currently agree.
- A guard on the split's guard: six seeds do not all produce one split, so a "deterministic" split
  that ignored the seed would not pass.
- The last test hands the built spec to `FakeTrainingProvider` — the first time M8-T01's port and
  M8-T02's builder meet.
- The golden is byte-identical: extracting `as_network_input` reordered nothing, verified against a
  literal copy of the pre-extraction code on three shapes and both polarities before the suite ran.

## References

- ADR-0015, ADR-0016, ADR-0023 — what a prepared picture is, decided for inference
- ADR-0078 / M7-T09 — the labels, and the split it deferred to here
- ADR-0080 / M8-T01 — `DatasetSpec`, and why it carries counts rather than only a path
- ADR-0006 — the training/inference separation `imaging` sits outside of
- ADR-0044 — `source`, and why the caller names the scope
- PROJECT_RULES §5 — `cache/` is safely deletable
