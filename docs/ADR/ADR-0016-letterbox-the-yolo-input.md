# ADR-0016 — The YOLO input is letterboxed, not squashed

- **Status:** Accepted
- **Date:** 2026-08-05
- **Affects:** `nanoscope/infrastructure/models/yolo.py` · audit **D-21** · M3-T04
- **Numerical impact:** none on a square scan — byte-identical, and the golden proves it on
  all 7 phantoms. On a non-square scan, every prepared pixel and every returned box moves.

## Context

`_prepare_image` resized any height map to `640 × 640` regardless of its shape, and
`_scale_boxes` mapped detections back with two different factors:

```python
img = cv2.resize(z_above, (self.yolo_size, self.yolo_size))          # squash
scale = np.array([w / 640, h / 640, w / 640, h / 640])               # unsquash, per axis
```

The two are consistent — a box comes back where it belongs — so the defect is not misplaced
detections. It is that **the model never saw the sample**. On a 256 × 512 scan the axes are
scaled by factors differing by 2×, so a circular particle is an ellipse of aspect 2 in model
space. Then `_boxes_to_detections` computes `radius_px = min(x2-x1, y2-y1) / 2`, which after
the anisotropic un-scaling is the smaller half-axis of that ellipse — reported as a radius,
converted to `radius_nm`, and measured against.

This is audit **D-21**, rated medium. It is rated medium because square scans are the common
case; on those the code is already correct and this ADR changes nothing.

## Decision

**Scale isotropically to fit, then pad to the square. Invert exactly that.**

```python
scale = self.yolo_size / max(h, w)
pad_x, pad_y = (self.yolo_size - round(w * scale)) // 2, (self.yolo_size - round(h * scale)) // 2
```

Three choices inside that:

**One helper computes the geometry, and both directions call it.** `_prepare_image` applies
the map, `_scale_boxes` inverts it. The defect being fixed is precisely a forward and an
inverse that disagreed, so they are not allowed to hold separate copies of the arithmetic. A
round-trip test fails if they ever do.

**The padding is applied after the normalisation, and its value is 255.** After the
min-max stretch and the inversion, 255 is what the *lowest* point of the map looks like — the
substrate. A border of 255 therefore reads as more substrate rather than as a step edge the
detector can respond to. Padding before the normalisation would let the border participate in
the min-max stretch and shift every grey level in the image, which is D-03 all over again in
a different disguise. ultralytics' own convention is 114 grey; that is a neutral choice for
photographs and the wrong one here, because in an inverted height map mid-grey means "a
particle of middling height", and we would be drawing a large one around the sample.

**The scale is a single float, and the inverse divides by it.** `round(w * scale)` can be
half a pixel away from `w * scale`, so an implementation could instead invert with the
realised integer dimensions and be marginally more exact per axis. It would also be
anisotropic again, by up to half a pixel. Given the defect this ADR exists to fix, exact
isotropy is worth more than half a pixel of fit, and the residual is far below the ±1 px the
box coordinates already carry from `int()` in `_boxes_to_detections`.

## Consequences

**Positive**

- Particles keep their shape, so `radius_px` is a radius on every scan shape, not only on
  square ones.
- The forward and inverse maps are one function apart and cannot drift.
- Non-square scans become usable at all, which matters because the Nanoscope parser does not
  require square data and the operator's `data/` directory is not guaranteed to be square.

**Negative**

- **The weights were trained on squashed images**, if the training set went through this same
  preparation. As with ADR-0015, a correct input is a distribution shift for a model that
  learned the incorrect one; this ADR makes the input right, it does not make the detections
  better, and nothing in the gate can measure which way it went (M3-T15, M7).
- A letterboxed non-square scan uses less of the model square than a squashed one did — a
  2:1 scan now fills half of it. That is the honest trade: the alternative is filling the
  frame with a distorted sample.
- `_prepare_image` is no longer three lines. It is nine, and two of them are the padding
  arithmetic.

**Neutral**

- **Square scans are byte-identical.** `max(h, w) == h == w` gives `scale = 640/side` and
  zero padding, which is the old code exactly. The characterization golden's 7 existing
  `yolo_input_preparation` blocks do not move, and that is asserted rather than assumed.
- Because every phantom is square, the harness could not see this fix at all. It gains
  `non_square_half_height` — the top half of each phantom, prepared — in the same commit, on
  the same reasoning as M3-T01's harness change: a fix that leaves its own path
  uncharacterized is not finished.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Pass the scan at native resolution and delete `_prepare_image`'s resize entirely — both backends accept any size and return boxes in the input's coordinates | Tempting, and it deletes code. But `MakeCropsDetectThem` resizes whatever it receives to a multiple of the crop size with a plain `cv2.resize`, which is the same anisotropic squash, now inside a dependency where no ADR of ours governs it | Only the direct backend existed |
| Pad with 114 grey, as ultralytics does | Mid-grey in an inverted height map is a particle of middling height. The convention comes from photographs, where it is neutral | The inversion were removed |
| Pad on one side (top-left) instead of centring | Same arithmetic, one fewer subtraction, and it puts the sample in a corner of the frame — a systematic position bias for no gain | — |
| Invert with the realised integer dimensions per axis | Marginally better fit, but anisotropic by construction, which is the defect | The rounding error ever exceeded the ±1 px the box `int()` already costs |
| Fix the single-crop tiling in the same commit | A different defect, found while reading for this one, and it changes what inference does rather than what it is fed. Filed as **M3-T21** | — |

## Compliance

- `tests/unit/test_yolo_input.py` — 5 tests for the geometry on top of the 6 for D-03: a
  circle on a 2:1 scan stays a circle, the border is exactly 255, an awkward 37 × 91 shape
  round-trips through forward-then-inverse, a square scan is not padded, and a 4:64 strip
  still produces the model square. Restoring the squash turns **4 of the 5 red**; the fifth
  is the square-scan invariant, which passes either way *by design* — it is what guarantees
  the golden does not move.
- Golden: the 7 existing blocks unchanged, one key added per phantom.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-21
- `ADR-0015` — the other defect in these lines, fixed one commit earlier
- `ADR-0010` (one defect, one commit, one ADR) — why M3-T21 is not in this commit
- **M3-T21** — `use_tiling=True` produces exactly one crop; found while reading for this task
