# CURRENT TASK

**ID:** `M3-T04`
**Title:** Aspect-ratio-preserving YOLO letterbox; isotropic box rescale
**Milestone:** M3 — Numerical correctness, third task
**Defect:** **D-21**, medium · **ADR:** **ADR-0016**
**Branch:** `sci/yolo-letterbox` (branched off `sci/yolo-normalise-then-cast`, which is
pushed and awaiting CI)
**Status:** **done 2026-08-05.** Rewritten for the next task at the start of the next
session; the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

Same three lines as M3-T03, the other half of "the detector is fed correctly".
`_prepare_image` resizes any scan to `640 × 640` regardless of shape, and `_scale_boxes`
undoes it with two different factors. On a 256 × 512 scan the x and y factors differ by 2×:
a circular particle is an ellipse in model space, and `radius_px = min(w, h) / 2` — the
number that becomes `radius_nm` — is the smaller half-axis of that ellipse, not a radius.

---

## What the reading turned up

Three things that were not in the task description and change what the fix has to be:

1. **Both backends return boxes in the coordinate space of the image handed to them.**
   `patched_yolo_infer` with `resize_initial_size=True` (the default, and what this code
   gets) maps crop detections back to `source_image` — the array we passed. ultralytics does
   the same. So `_scale_boxes` is the exact inverse of our own resize, and nothing else.
2. **Handing over the native-resolution scan instead does not fix D-21.** `MakeCropsDetectThem`
   resizes whatever it receives to a multiple of the crop size with `cv2.resize(image, (x_new,
   y_new))` — anisotropically. The squash would move into the library, out of reach. So the
   image we pass must already be square; letterboxing is the right fix, not deletion.
3. **`use_tiling=True`, the default, currently produces exactly one crop.** With a 640 × 640
   input and `shape_x = shape_y = 640`, `get_crops_xy` computes
   `int((640-640) / (640*0.75)) + 1 = 1` step on each axis. The sliding window covers the
   whole image in a single tile, so the tiled backend does the same work as the direct one,
   more slowly, and the reason it exists — small particles seen at native resolution — never
   happens. The letterbox does **not** fix this (the input is still 640 × 640). It is a
   separate defect and gets filed, not fixed here.

---

## Scope

**In scope**

1. `_prepare_image`: isotropic resize to fit, then pad to `yolo_size` square
2. `_scale_boxes`: subtract the padding, divide by the one scale factor
3. One geometry helper shared by both, so the forward and inverse maps cannot drift apart
4. **A non-square case in the characterization harness.** Every phantom is square
   (256 × 256, one 128 × 128), so this fix is invisible to the golden as it stands — the
   defect it repairs is not characterized at all. Same reasoning as M3-T01's harness change
5. Tests, and **ADR-0016** — including what the padding value is and why

**Out of scope**

- The single-crop tiling finding above — filed as **M3-T21**, its own task
- **D-09 / M3-T05** (confidence), **M3-T18** (`_last_result` typing)
- Changing `yolo_size`, the backends, or anything about inference

---

## Definition of done

- [x] Isotropic scale in both directions; padding accounted for in the inverse
- [x] A square input produces byte-identical output to today — the M3-T03 baseline must not
      move, and the golden proves it on all 7 phantoms
- [x] A non-square input keeps circles circular, and a box round-trips to its own coordinates
- [x] Harness records a non-square preparation, so the fixed path is characterized
- [x] `make check` green — 136 tests; golden drift is 7 ADDED keys and nothing else; golden drift confined to the new key
- [x] **M3-T21 filed** with the arithmetic that proves it
- [x] ADR-0016; `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M3-T04: letterbox the YOLO input instead of squashing it`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| The padding reads as image content and produces detections in the border | Pad with the value that means "substrate" after inversion — the same bright level the background already has, so the pad is not an edge the detector can latch onto. Named in the ADR |
| Padding is included in the min-max normalisation and shifts every grey level | Pad **after** normalising. The stretch is computed from real data only |
| Forward and inverse maps drift apart in a later edit | One helper computes both; a round-trip test fails if they disagree |
| The golden shows nothing and the fix looks unverified | It shows nothing *because every phantom is square* — that is the finding, and the harness gains the non-square case in the same commit |
