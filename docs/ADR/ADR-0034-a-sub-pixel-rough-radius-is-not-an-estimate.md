# ADR-0034 — A rough radius below one pixel is not an estimate

- **Status:** Accepted
- **Date:** 2026-08-07
- **Affects:** `nanoscope/core/science/preprocessing/substrate.py` · **B-061** · M3-T23
- **Numerical impact:** **11 golden differences, all inside one cell** —
  `afm_sparse_low_snr.build_substrate_map_no_scale.sizes`. The object count falls **3351 → 627**
  and the Otsu threshold moves **7.7e-09 → 1.46**. The returned `substrate`, `z_above` and
  `opening_radius` are **unchanged**.

## Context

`estimate_rough_radius` could return **0**, and `get_substrate_map(z, 0)` opens with `disk(0)` — a
single pixel — so **the opening is the identity**. The "substrate" comes back equal to the image
and `z_above` is zero everywhere. Nothing raises, nothing warns, and the result has the shape of
an answer.

Measured across the phantom set, the cell that reaches it:

| Phantom | scale | median object area | rough radius | rough opening |
|---|---|---:|---:|---|
| `afm_flat_monodisperse` | either | 234.5 px | 14 | real |
| `afm_tilted_polydisperse` | either | 186.0 px | 12 | real |
| `afm_dense_overlapping` | either | 144.5 px | 11 | real |
| `afm_coarse_pixels` | either | 73.5 px | 7 | real |
| `afm_sparse_low_snr` | scaled | **1.0 px** | 3 | real |
| **`afm_sparse_low_snr`** | **unscaled** | **1.0 px** | **0** | **identity** |

The two bottom rows share the same worthless input: `median + std` selected **single-pixel
noise**, so the median object area is 1 px. The scaled run survives only because
`min_size_px = 5 / 1.95 = 2.56` floors the answer — the estimate is equally worthless there and
the floor hides it. Without a scale there is no floor, correctly (ADR-0025), so the worthless
estimate goes through as zero.

### This corrects ADR-0025's diagnosis

ADR-0025 recorded, on this exact phantom and path, **17 objects → 3351**, and explained it as
*"losing the scale is losing the filter"*. True, and not the whole mechanism. Losing the scale did
**two** things:

1. it skipped the `min_size_nm` filter — the effect that ADR named;
2. it collapsed the rough radius to zero, so Otsu ran on a map that **had never been opened**.

Fixing only the second moves **3351 → 627**. The collapsed radius accounted for roughly four
fifths of the inflation; the missing filter for the rest.

## Decision

**A rough radius below one pixel means the estimate failed, and takes the fallback the function
already has.**

```python
if rough_radius < 1.0:
    logger.warning("the rough radius estimate came out sub-pixel (%.3g px) because the median "
                   "object found is %.3g px in area — that is noise, not a particle; falling "
                   "back to 1%% of the image width", rough_radius, median_area)
    return _fallback_radius(z, min_size_px)
```

The `len(props) == 0` branch already answers "the image is too flat or too noisy" with 1 % of the
image width. A median object of one pixel *is* that situation, arriving by a different route, so
it takes the same exit — and the two warnings say which route it was, because an image with
nothing in it and an image full of noise are different problems with the same answer.

**Checked before the rounding.** `_integer_radius` ceils, and `ceil(0.96)` is 1, so a check after
it would only ever catch an exact zero — the symptom rather than the condition.

**The fallback is right where it can be checked.** On the unscaled sparse phantom it returns
**3 px**, which is exactly what the *scaled* run of the same image computes. The one case with a
known-good answer agrees with it.

**Not raised, and `get_substrate_map(z, 0)` stays legal.** The image is valid and the automatic
path is documented to fall back; refusing it would be ADR-0017's case, not ADR-0018's. M3-T13
deliberately left a zero radius accepted at the funnel, and this ADR does not revisit that — it
removes the only route that produced one.

## Consequences

**Positive**

- The rough opening is always an opening. A substrate identical to the image is no longer
  reachable through the automatic path.
- The unscaled path costs far less than ADR-0025 measured, and the remaining cost is attributable
  to the one cause that is genuinely about scale.
- Two failure routes, one fallback, two distinguishable messages.

**Negative**

- A number in the golden moves, and it is a number an earlier ADR quoted. ADR-0025 is not edited
  — accepted ADRs are immutable (ADR index rules) — so its 3351 stands with this ADR as the
  correction, which is what the supersession discipline is for.
- The fallback is still `1 %` of the image width, a constant nobody has justified. It was already
  there; this commit gives it a second caller and no new authority. If it is wrong, it is wrong in
  two places now.

**Neutral**

- Four AFM phantoms and both image phantoms do not move: their estimates are several pixels and
  never approach the branch.

## What is deliberately not in this commit

**`radius_px = int(np.sqrt(median_area / np.pi))` — filed as B-063.** It is a second, undeclared
rounding inside a function whose only rounding is supposed to be `_integer_radius` (ADR-0020), and
it is the same `int()` pattern ADR-0024 deleted as D-04's mechanism. It is also *how* the estimate
reaches exactly zero rather than 0.96.

Deleting it moves the rough radius on **every** phantom — measured: 14 → 15, 12 → 14, 11 → 12,
7 → 9 — and therefore the final radius, the substrate and every height in the golden. That is a
different blast radius and a different review; bundling it here would make neither attributable
(ADR-0010).

**B-062** — recall 0.000 on the same phantom's *detection*. Different function, and it wants an
operator's view of the sensitivity trade-off.

**The `median + std` threshold.** That it selects noise on a low-SNR scan is the input to this
decision, not a defect being fixed here.

## The measured delta

**11 differences, every one inside `afm_sparse_low_snr.preprocessing.build_substrate_map_no_scale.sizes`.**
No other phantom, no other block, no other path.

| Quantity | Before | After |
|---|---:|---:|
| `n_objects_reported` / `n_radii_kept` | 3351 | **627** |
| `otsu_threshold` | **7.7e-09** | 1.459 |
| `radii_px.max` | 6.68 | 93.52 |
| `radii_px.sum` | 3739.6 | 662.7 |
| `radii_px` 75th / 90th percentile | 1.38 / 2.19 | 0.98 / 1.49 |
| `opening_radius` · `substrate` · `z_above` · `typical_radius_px` | — | **unchanged** |

**The Otsu threshold of 7.7e-09 is the fingerprint of the defect**, and it had been in the golden
all along. It is Otsu applied to an all-zero map — which is what `z - z` is when the opening is
the identity. The harness recorded a threshold of numerical zero and 3351 objects on a phantom
with six particles, and it read as normal, because nothing in the file says what a plausible
value looks like. This is the opposite of the last four findings: the golden *had* the evidence.

**What did not move matters as much.** `opening_radius` stays 5 and the returned `substrate` and
`z_above` are byte-identical, because the *median* radius — which is what drives the final radius
— is 0.798 px before and after. The fix changes what the function **reports about the sample**,
not what it **returns as the substrate**, on this image. That is a coincidence of this median, not
a property: on an image whose median moves across the `× 2.5` boundary, the substrate would move
too.

**And it does not make the unscaled run good.** The median radius is still 0.798 px — noise — and
`radii_px.max` rises to 93.5 px because with a real opening one large background component now
clears the threshold. 627 objects on a phantom with six particles is still wrong; it is wrong for
the reason ADR-0025 named, with the compounding defect removed. The remainder belongs to
**B-062**.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Floor the radius at 1 px | `disk(1)` is a 3×3 element: it removes single-pixel noise and nothing else, so Otsu still measures noise and the estimate stays wrong — a smaller lie, told quietly | The input were merely small rather than meaningless |
| Raise | The automatic path has a documented fallback and the image is valid data. ADR-0018 settled that shape of case | The caller had asked for something the data cannot supply |
| Reject a zero radius in `get_substrate_map` instead | Treats the symptom at the funnel and leaves the estimator returning a number it does not have. M3-T13 considered exactly this and filed the question rather than moving a number inside a validation task | There were no automatic path — only radii the caller chose |
| Return `None` for "no estimate" | Honest, and it makes every caller handle a case the function already has a documented answer for. The fallback *is* the answer | The callers could act differently on "no estimate" |
| Fix the `int()` truncation at the same time | It is the proximate cause, and it moves every phantom. Two deltas in one commit is the thing ADR-0010 exists to prevent | The truncation affected only this cell |

## Compliance

- `tests/unit/test_rough_radius.py` — **9 tests**: the estimate is never sub-pixel on a pure-noise
  map; the opening it produces is therefore never the identity; `disk(0)` *would* still be the
  identity, pinned so the reason for the guard cannot quietly stop being true; the fallback is 1 %
  of the width at two image sizes; each of the two routes into that fallback says which one it
  was; an image with real particles never reaches the branch and is identical with and without a
  scale; and the substrate consequence, which asserts the unscaled run still differs from the
  scaled one — the claim M3-T20's test makes and this must not undo. **Removing the branch turns
  5 of the 9 red.**
- The full suite is green with M3-T20's `test_and_that_costs_the_substrate_on_a_noisy_scan`
  unchanged: the unscaled run still counts more objects and still uses a different opening radius.
- Golden: the eleven differences above.

## References

- `ADR-0025` — the finding this corrects, and the reason there is no floor without a scale
- `ADR-0020` — `_integer_radius` is the only rounding a radius gets; see also **B-063**
- `ADR-0017` / `ADR-0018` — when the analysis raises and when it answers
- **B-061**, closed here; **B-063**, filed here
