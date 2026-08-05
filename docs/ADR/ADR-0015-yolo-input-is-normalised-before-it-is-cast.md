# ADR-0015 — YOLO input is normalised before it is cast to `uint8`

- **Status:** Accepted
- **Date:** 2026-08-05
- **Affects:** `nanoscope/infrastructure/models/yolo.py` · audit **D-03** · M3-T03
- **Numerical impact:** every YOLO detection ever produced by this repository. The prepared
  image changes on all 7 phantoms; on one of them the new image is **anti-correlated**
  (r = −0.499) with the old one.

## Context

`YoloDetector._prepare_image` turns a height map into the 8-bit RGB image the network
consumes. It did the two steps in the wrong order:

```python
img = cv2.resize(z_above, (640, 640)).astype(np.uint8)   # cast first — truncates
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)  # too late, the data is gone
```

`z_above` is a float height map in nanometres. Casting it to `uint8` first keeps only
whichever integers between 0 and 255 happen to fall inside its range, and C-style wraps
everything outside: `uint8(260) == 4`, `uint8(-1) == 255`. Normalising afterwards stretches
what is left, which makes the result *look* like a well-exposed image.

Measured on the characterization phantoms, distinct grey levels reaching the network:

| phantom | height range (nm) | levels, cast-first | levels, normalise-first | corr. |
|---|---|---:|---:|---:|
| `afm_flat_monodisperse` | −0.5 … 18.1 | 19 | 256 | 0.997 |
| `afm_tilted_polydisperse` | −1.8 … 45.6 | 47 | 255 | 0.914 |
| `afm_dense_overlapping` | −0.6 … 19.0 | 19 | 256 | 0.997 |
| `afm_sparse_low_snr` | −4.3 … 5.0 | 8 | 239 | **−0.499** |
| `afm_coarse_pixels` | −0.6 … 20.0 | 21 | 256 | 0.997 |
| `sem_bright_particles` | 14.5 … 230.6 | 208 | 255 | 1.000 |
| `tem_dark_particles` | 23.7 … 234.3 | 200 | 254 | 1.000 |

Two things that table shows and a single summary number would hide:

- **The damage scales with how good the data is.** A quiet AFM scan with a 5 nm range —
  which is a *well-prepared sample*, not a bad measurement — lost the most: 8 levels of 256,
  and the negative heights wrapped to near-white, so the resulting image is closer to the
  negative of the correct one than to the correct one. The SEM/TEM phantoms, whose values
  already span most of 0–255 because they are images rather than physical heights, barely
  moved.
- **Monotonicity was destroyed, not just resolution.** The audit's probe
  `[-10, -1, 0, 1, 5, 100, 260, 300]` becomes `[246, 255, 0, 1, 5, 100, 4, 44]`: the tallest
  feature in the map becomes one of the darkest pixels. Any monotone transfer function would
  be survivable — a detector can learn around a gamma curve. This one is not monotone.

This is audit **D-03**, rated critical, and audit §5 R6 requires it to ship alone.

## Decision

**Normalise in floating point, then cast.** The resize is unchanged; the cast moves to after
the normalisation:

```python
img = cv2.resize(z_above, (self.yolo_size, self.yolo_size))
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
img = cv2.bitwise_not(img)
```

Two smaller choices inside that decision:

**The cast truncates (`.astype(np.uint8)`) rather than rounds.** `cv2.normalize(...,
dtype=cv2.CV_8U)` would round instead, and rounding is marginally more faithful. Truncation
is chosen because it is exactly the reference implementation the characterization harness has
compared against since the baseline was recorded, which makes the golden's
`mean_abs_diff_vs_normalize_first` land on **0.0** — the defect's own measuring stick now
reads zero. The difference between the two is at most one grey level in 256; it is below the
resolution of anything downstream, and picking the one that makes the fix legible in the
golden is worth more than 0.4% of a level.

**Min-max stays the normalisation.** It is what the code already did and what the recorded
reference assumed. It means preparation is invariant under `z → a·z + b` for `a > 0` — the
same scan reported in ångström or with a different zero produces the identical input — and
that property is now asserted by a test. A percentile or fixed-window normalisation would be
a different decision, with a different failure mode on outliers; nothing in the task requires
one.

## Consequences

**Positive**

- The network sees the data. Between 3.1% and 81.2% of the dynamic range survived
  preparation before; 100% survives now.
- Height order is preserved for every input, including maps taller than 255 nm, which
  previously wrapped.
- Preparation no longer depends on the physical units of the map.
- The golden's D-03 measurement (`mean_abs_diff_vs_normalize_first`) becomes a permanent
  regression guard reading 0.0 instead of a defect size.

**Negative**

- **Every stored YOLO detection predates this fix and was produced from corrupted input.**
  Radii, counts and any benchmark run through this path are not comparable across the fix.
  Nothing in the repository stores such results today — there is no project database yet
  (M4) — so the cost is borne by the operator's own notebooks and exports, not by code.
- **The weights in `checkpoints/best12x.pt` were trained on images prepared by the old
  path.** If the training set was built with this same function, the model has learned to
  read 8-to-47-level posterised images, and feeding it correct ones is a distribution shift
  that can lower detection quality *before* it raises it. This ADR does not resolve that; it
  makes the input correct, which is the precondition for retraining (M7) or for an honest
  evaluation (M3-T15). **Do not read this fix as "detections are now better" — read it as
  "detections are now computed from the data".**
- The gate cannot confirm any of that. Inference stays outside the golden by
  PROJECT_RULES §6, so what is proven here is the preparation stage alone.

**Neutral**

- Constant maps are unchanged: `max == min`, there is no range to stretch, and both orders
  produce a uniform image. Recorded because §6 requires degenerate inputs to have an answer.
- Nothing outside `_prepare_image` moved. Both backends call it, so both are fixed by the
  same three lines.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| `cv2.normalize(..., dtype=cv2.CV_8U)` — normalise and cast in one call, rounding | Correct, and 1/256 more faithful, but it diverges from the reference the golden has recorded since the baseline, so the fix would land as "a new number" instead of "the distance to correct is now zero" | The harness reference is ever re-derived; then take the rounding version |
| Keep `float32` all the way into the model | ultralytics and `patched_yolo_infer` both expect an 8-bit RGB array here; changing that is a provider-boundary change, not a defect fix | The provider port (M4) makes the input dtype part of the contract |
| Percentile normalisation (e.g. 1–99%) to resist outliers | A real improvement on scans with a spike, and a *different* decision with its own ADR. Bundling it would make this fix unattributable — ADR-0010 | Evaluation (M3-T15) shows outliers dominate real scans |
| Fix the aspect-ratio distortion in the same commit | That is **D-21 / M3-T04**, a separate defect in the same three lines. One commit, one intent | — |
| Retrain the weights in this task | Not a numerical-correctness task; needs the training pipeline (M7) and an evaluation harness (M3-T15) to say whether it helped | — |

## Compliance

- `tests/unit/test_yolo_input.py` — 6 tests stating properties of the mapping
  height → grey level: full dynamic range, sub-unit ranges, monotonicity, no wraparound past
  255, invariance under affine rescaling, and the constant-map degenerate case. Restoring the
  old order turns **5 of the 6 red**.
- The golden moves on all 7 phantoms under `yolo_input_preparation`, and only there.
- Inference is not tested and is not claimed to be tested.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-03, §5 R6, §6 (why inference is uncharacterized)
- `ADR-0008` (the golden is the contract) · `ADR-0010` (one defect, one commit, one ADR)
- `ADR-0014` — the first numerical fix; this is the second
- **M3-T04 / D-21** — the other defect in these lines: `cv2.resize` to a square destroys the
  aspect ratio, and `_scale_boxes` un-destroys it anisotropically
