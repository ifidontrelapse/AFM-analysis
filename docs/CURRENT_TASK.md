# CURRENT TASK

**ID:** `M3-T23`
**Title:** A rough radius that lands below one pixel is not an estimate
**Milestone:** M3 — Numerical correctness, twenty-first task
**Defect:** **B-061** (filed by M3-T13, which found it by being too strict) · **ADR:** **ADR-0034**
**Branch:** `sci/m3-numerical-correctness` (the consolidated branch — see the declared
deviation from PROJECT_RULES §7 in `STATE.md`)
**Status:** **done 2026-08-07.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M3's remaining items are `M3-T16` (blocked on **B6**), `M3-T19` (`low`, a typing finding),
**B-061** and **B-062**. B-062 is a detector-tuning question with a trade-off the operator should
see; B-061 is not a question at all once the mechanism is measured — the function returns a
number that means "I could not estimate", and every caller reads it as a radius.

## The defect, measured

`estimate_rough_radius` can return **0**. `get_substrate_map(z, 0)` opens with `disk(0)`, a single
pixel, so **the opening is the identity**: the "substrate" comes back equal to the image and
`z_above` is zero everywhere. It looks exactly like a result.

Measured across the phantoms — the cell that reaches it:

| Phantom | scale | median object area | rough radius | rough opening |
|---|---|---:|---:|---|
| `afm_flat_monodisperse` | either | 234.5 px | 14 | real |
| `afm_tilted_polydisperse` | either | 186.0 px | 12 | real |
| `afm_dense_overlapping` | either | 144.5 px | 11 | real |
| `afm_coarse_pixels` | either | 73.5 px | 7 | real |
| `afm_sparse_low_snr` | scaled | 1.0 px | 3 | real |
| **`afm_sparse_low_snr`** | **unscaled** | **1.0 px** | **0** | **identity** |

The median object area is **1.0 px** in both of the last two rows: the threshold
`median + std` found single-pixel noise, not particles. The scaled run survives it only because
`min_size_px = 5 / 1.95 = 2.56` floors the answer — **the estimate is equally worthless there and
the floor hides it.** Without a scale there is no floor (ADR-0025, correctly), so the worthless
estimate goes through as 0.

### It corrects ADR-0025's diagnosis

That ADR recorded, for this exact phantom and path, **17 objects → 3351** and explained it as
*"losing the scale is losing the filter"*. That is true and it is not the whole mechanism. Losing
the scale did **two** things: it skipped the `min_size_nm` filter, and it collapsed the rough
radius to zero so Otsu ran on a map that had never been opened. Fixing only this half moves
**3351 → 627**, so the collapsed radius accounted for roughly **four fifths** of the inflation
and the missing filter for the rest.

---

## The decisions this task has to make

### 1. What a sub-pixel estimate means

| | |
|---|---|
| **It means the estimate failed** ✅ | A median object area of ~1 px is noise, and the function already has a branch for exactly that situation — `len(props) == 0` warns "too flat or too noisy" and falls back to 1 % of the image width. This is the same condition arriving by a different route |
| Floor it at 1 px | `disk(1)` is a 3×3 element: it removes single-pixel noise and nothing else, so Otsu still measures noise and the estimate stays wrong — a smaller lie, quietly |
| Raise | The image is valid and the automatic path has a documented fallback. ADR-0018's case, not ADR-0017's |

The fallback lands on **3 px** for the unscaled sparse phantom — which is what the *scaled* run of
the same image computes. The one case that can be checked against a known-good answer agrees.

### 2. Where the check goes

On the **unrounded** radius, before `_integer_radius` ceils it. `ceil(0.96)` is 1, so a check
after the rounding would never see the sub-pixel case at all — it would only ever catch an exact
zero, which is the symptom rather than the condition.

### 3. What this task does *not* touch

`radius_px = int(np.sqrt(median_area / np.pi))` is a **second, undeclared rounding** in a function
whose only rounding is supposed to be `_integer_radius` (ADR-0020), and it is the same `int()`
pattern ADR-0024 deleted as D-04's mechanism. It is also *how* this estimate reaches exactly zero.

But removing it moves the rough radius on **every** phantom — measured: 14 → 15, 12 → 14,
11 → 12, 7 → 9 — and therefore the final radius, the substrate and every height. That is a
different defect with a different blast radius, and bundling it here would make neither
attributable. **Filed as B-063 with those numbers.**

---

## Scope

**In scope**

1. A rough estimate below 1 px falls back, with a warning that names the median area it rejected
2. Tests: the reproduction, the fallback value, the untouched phantoms, and that the warning
   distinguishes this case from the empty one
3. The golden re-recorded for the one cell that moves

**Out of scope**

- **B-063** — the `int()` truncation, filed with its measured effect on all five phantoms
- **B-062** — recall 0.000 on the same phantom's *detection*. Different function, different
  decision, and it wants an operator's view of the sensitivity trade-off
- The `median + std` threshold itself. That it finds noise on a low-SNR scan is the input to this
  decision, not a defect this task is fixing

---

## Expected blast radius, before measuring

- **One golden cell**: `build_substrate_map_no_scale` on `afm_sparse_low_snr`. `n_objects`
  3351 → 627, and everything derived from the radii with it. The opening radius stays 5, so the
  unscaled run still differs from the scaled one — ADR-0025's finding survives with a smaller
  number and a sharper cause.
- **Four AFM phantoms and both image phantoms: nothing.** Their estimates are not sub-pixel.
- M3-T20's `test_and_that_costs_the_substrate_on_a_noisy_scan` must still pass: it asserts the
  unscaled run differs from the scaled one, which remains true.

---

## Definition of done

- [x] A sub-pixel rough estimate falls back and says so, naming the median area
- [x] Tests — **9**; removing the branch turns 5 red
- [x] `make check` green — 434 tests; delta **11 differences, all inside the one predicted cell**
- [x] ADR-0034; **B-063 filed** with measurements; `Backlog.md` (B-061 → done), `STATE.md`,
      `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [x] Commit: `M3-T23: a rough radius below one pixel is not an estimate`

---

## What it turned up

**The golden had the evidence and nobody read it.** An Otsu threshold of **7.7e-09** is Otsu
applied to an all-zero map — precisely what `z − z` is when the opening is the identity. It sat
in the baseline beside "3351 objects" on a phantom with six particles. The four previous findings
in this milestone were the harness *failing to record* something; this is the first where it
recorded the fingerprint and the number meant nothing to a reader.

**The substrate did not move, and that is luck.** `opening_radius` stays 5 because the *median*
radius is 0.798 px before and after. On an image whose median crosses the `× 2.5` boundary the
returned substrate would move too — so this is not evidence that the defect was harmless.

**The scaled run was equally broken and better hidden.** Same image, same worthless 1.0 px median
object; the only thing that saved it was the `min_size_nm` floor. A floor is not an estimate.

---

## Notes

The audit never listed this. It was found by M3-T13 writing a validation rule that was *too
strict* — `ensure_positive(radius_px)` turned an M3-T20 test red — and the honest response then
was to relax the check and file the question. Three tasks later it is answered with a number.
