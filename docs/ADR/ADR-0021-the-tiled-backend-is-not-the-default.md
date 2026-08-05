# ADR-0021 — The tiled YOLO backend is not the default

- **Status:** Accepted
- **Date:** 2026-08-05
- **Affects:** `nanoscope/infrastructure/models/yolo.py` · `nanoscope/core/entities/pipeline.py` ·
  M3-T21 · decision **B7**
- **Numerical impact:** **none recorded.** Inference is outside the gate (PROJECT_RULES §6), so
  the golden neither recorded the tiled backend before nor records the direct one now. The
  behaviour change is real and is stated in Consequences.

## Context

`YoloDetector` has two backends. The tiled one, `use_tiling=True`, was the default and has
**never tiled**:

```python
# _prepare_image returns exactly one yolo_size square — 640 x 640
# MakeCropsDetectThem is given shape_x = shape_y = yolo_size = 640
# get_crops_xy: int((640 - 640) / (640 * 0.75)) + 1  ==  1 step per axis
```

One crop, covering the whole image. The tiled backend therefore ran the direct backend's work
through an extra library, more slowly, and the thing tiling exists for — seeing small particles
at native resolution instead of downscaled into a 640 px square — never happened.

The overlap is not the lever: `int((side − shape) / step) + 1` is 1 for *any* step when
`side == shape`. Only the input size is. Real tiling needs `shape * (2 − overlap/100)` =
**1120 px** at the current settings, and a 512 px scan cannot reach it without being upsampled
first.

That is what made this a decision rather than a bug fix. The three ways out cost different
things:

- **Upsample the scan to ≥ 1120 px, then tile.** The model then examines interpolated pixels —
  detail that was never measured.
- **Use a crop shape below 640.** Tiling starts working, and the model upscales each crop
  instead; inference cost rises roughly with the number of crops.
- **Accept that tiling does nothing at this resolution.**

## Decision

**`use_tiling` defaults to `False`, in both `YoloDetector` and `PipelineConfig`. The tiled
backend is kept, not deleted.**

The default now describes what actually happens. Nothing about the pipeline's output changes on
a 512 px scan, because both backends were doing the same single-crop inference — the direct one
simply does it without a sliding-window library in the way.

**Kept, because the fix is an input-size question and nobody can answer it yet.** Choosing
between "upsample and tile" and "smaller crops" is a resolution-versus-cost trade-off, and the
project has no way to measure detection quality: **M3-T15**, the evaluation harness that scores
precision, recall and localisation against phantom ground truth, does not exist. Deleting the
backend now would mean rewriting it later from a git history; leaving it costs one branch.

**Asking for tiling anyway logs what it will do.**

```python
def _warn_if_single_crop(self, img) -> bool:
    if self._crop_steps(img.shape[1]) * self._crop_steps(img.shape[0]) > 1:
        return False
    logger.warning("tiling requested but the %dx%d input is one %d px crop: ...")
```

It is a separate method for one reason: `_detect_tiled` imports `patched_yolo_infer` on its first
line and then runs a model, so a test that went through it would need weights and would run
inference inside the gate. The guard is testable on its own; the arithmetic that produced this
whole finding is now a method, `_crop_steps`, with a test rather than a comment.

## Consequences

**Positive**

- The default no longer promises tiling that does not occur.
- The direct backend removes `patched_yolo_infer` from the default path, and with it a
  dependency's own `cv2.resize` — the one M3-T04 (ADR-0016) noted would silently reintroduce the
  squash if our resize were simply deleted.
- The single-crop degeneracy is now self-reporting, and goes quiet the day the input grows.

**Negative**

- **The two backends are not bit-identical**, even at one crop: `MakeCropsDetectThem` applies its
  own preprocessing and `CombineDetections` runs a second NMS pass at `nms_threshold=0.25` on top
  of ultralytics' `iou=0.7`. So detections on real scans may differ slightly. Nothing in the gate
  can see this — inference is outside it — and no claim is made here that either result is better.
  **M3-T15 owns that question.**
- `yolo_use_tiling=True` remains reachable and remains pointless until the input grows. The
  warning is the mitigation.

**Neutral**

- Zero golden difference, and for a stated reason rather than by luck: only `_prepare_image` is
  recorded, and it is untouched.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Upsample the scan to ≥ 1120 px and tile properly | The model would examine interpolated pixels; on a 512 px scan more than half of what it sees would be invented | The scans were natively large |
| Crop shape below 640 (e.g. 320) | Tiling would work and small particles would get more effective resolution — the genuinely promising option — but inference cost rises with crop count and nothing here can yet measure whether detections improve | M3-T15 existed and showed a gain |
| Delete the tiled backend | Tempting, and the shortest diff. But the decision is about input size, not about the backend; deleting it means writing it again from history the day the answer is "smaller crops" | M3-T15 showed no gain from any tiling configuration |
| Leave the default and only document it | A default that describes something the code does not do is the defect, not the documentation | — |

## Compliance

- `tests/unit/test_tiling.py` — 9 tests (4 parametrised): the default is the direct backend in
  both the detector and `PipelineConfig`; one crop at the prepared size, for every overlap
  (0/25/50/75), which is what makes "the overlap is not the lever" a fact; 1120 px tiles and
  1119 px does not; the warning fires on a degenerate input and stays silent on a genuine one.
- Golden: unchanged, by design.

## References

- `docs/audit/2026-07-28-baseline-audit.md` — M3-T21 was filed from reading for M3-T04, not by
  the original audit
- `ADR-0016` — `_prepare_image`, and why deleting our own resize would move the squash into a
  dependency
- **M3-T15** — the evaluation harness that has to exist before the crop-size question can be
  answered
- Decision **B7**, answered by the operator 2026-08-05
