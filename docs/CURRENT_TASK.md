# CURRENT TASK

**ID:** `M3-T24`
**Title:** The rough estimate stops truncating its own radius
**Milestone:** M3 — Numerical correctness, twenty-second task
**Defect:** **B-063** (filed by M3-T23, which fixed its consequence) · **ADR:** **ADR-0035**
**Branch:** `sci/m3-numerical-correctness` (the consolidated branch — see the declared
deviation from PROJECT_RULES §7 in `STATE.md`)
**Status:** planned — no code written yet.

---

## Why this task is next, and why it is not an operator decision

`M3-T16` is blocked on **B6** and **B-062** wants an operator's view of a sensitivity trade-off.
B-063 does not: **the rule was already decided**, and this is a site that was missed.

- **ADR-0020** ruled that every radius reaching `disk()` is `ceil`-ed — *"up, not down: a radius
  smaller than a particle recovers a substrate containing the particle"* — and that
  `_integer_radius` is the one funnel. `int()` inside the estimator is a **second, undeclared
  rounding**, in the opposite direction.
- **ADR-0024** deleted this exact `int()` pattern as D-04's mechanism.
- The parameter's own documentation, from the commit that introduced both, says `scale=1.7` is a
  *"multiplier so the disk is safely **larger** than a particle"*. The truncation makes it
  smaller. It works against the stated purpose of the line above it.

So removing it executes an accepted decision rather than making a new one.

## The defect

```python
radius_px = int(np.sqrt(median_area / np.pi))   # <- truncates, downward, silently
rough_radius = max(radius_px * scale, min_size_px)
return _integer_radius(rough_radius)            # <- the declared rounding, upward
```

Two roundings in three lines, in opposite directions, and only the second one is documented. The
first also loses up to a full pixel *before* the `× 1.7`, so the error is amplified: at
`sqrt(area/π) = 4.9` the disk is built from 4, not 4.9, and the ×1.7 turns a 0.9 px truncation
into a 1.5 px shortfall.

It is also how M3-T23's zero arose: `int(0.56) = 0`, and `0 × 1.7 = 0`.

---

## What it moves — measured before deciding

**M3-T23's own filing overstated this**, and the correction belongs here rather than in a quiet
edit. It said the change "moves the final radius, the substrate and every height". Measured:

| Phantom | rough | final | substrate | heights | measured rows |
|---|---|---|---|---|---|
| `afm_flat_monodisperse` | 14 → **15** | 19 → 19 | identical | identical | 24 → 24 |
| `afm_tilted_polydisperse` | 12 → **14** | 18 → **19** | **differs** | **max 0.686 nm (2.77 %)** | 30 → 30 |
| `afm_dense_overlapping` | 11 → **12** | 16 → 16 | identical | — | **59 → 60** |
| `afm_sparse_low_snr` | 3 → 3 | 8 → 8 | identical | identical | 0 → 0 |
| `afm_coarse_pixels` | 7 → **9** | 11 → 11 | identical | identical | 14 → 14 |

**The two-stage design absorbs most of it.** The rough radius moves on four phantoms; the *final*
radius moves on one, because the final radius comes from Otsu's median radius on the roughly
opened map and that median is robust to the opening being slightly larger.

**`afm_dense_overlapping` gains a measured particle without its substrate changing**, which is
worth understanding before it is recorded: `sizes` from the rough stage feeds
`estimate_log_params`, so the sigma range the LoG detector searches moves even when `z_above` does
not. One more particle clears it — 59 rows become 60.

### And detection quality, which nothing could measure before M3-T15

| Phantom | recall | mean localisation |
|---|---|---|
| `afm_flat_monodisperse` | 1.000 → 1.000 | 0.4310 → 0.4310 px |
| `afm_tilted_polydisperse` | 1.000 → 1.000 | 0.6137 → **0.6156** px |
| `afm_dense_overlapping` | 0.843 → 0.843 | 0.8265 → 0.8265 px |
| `afm_coarse_pixels` | 1.000 → 1.000 | 0.4143 → 0.4143 px |

**No measurable change in detection quality.** That is the honest reading and it is worth stating
plainly: this task is not an improvement to detection, it is the removal of an undeclared
rounding that contradicts an accepted ADR. **The first task in this project able to say that with
a number rather than an assumption** — which is what M3-T15 was built for.

---

## The decision this task has to make

Only one, and it is narrow: **is `sqrt(median_area / π)` allowed to keep its fractional part?**

| | |
|---|---|
| **Yes — `_integer_radius` is the only rounding** ✅ | ADR-0020's rule, applied where it was missed. The estimate stays a float until the single declared ceiling |
| Round it to nearest instead | Still a second rounding, still undeclared, and still ahead of a `× 1.7` that amplifies it |
| Keep it, document it | Documents a contradiction rather than removing it, and leaves a downward rounding inside a parameter whose purpose is an upward margin |

**What is not decided here:** whether `1.7` and `2.5` are the right constants. They are
undocumented beyond "safely larger", they were chosen with the truncation in place, and nothing
in the project can currently say what they should be — **M3-T15 can now measure a candidate**,
which makes it a real task rather than a preference. **Filed as B-064.**

---

## Scope

**In scope**

1. Delete the `int()`; the estimate reaches `_integer_radius` as a float
2. Tests: the truncation is gone, the two roundings become one, and the direction is up
3. The golden re-recorded — 4 phantoms' rough radius, 1 phantom's substrate and heights,
   1 phantom's row count

**Out of scope**

- **B-064** — the provenance of `1.7` and `2.5`
- **B-062** — recall 0.000 on `afm_sparse_low_snr`, which this does not move (3 → 3)
- `M3-T19` — the `low` mypy finding in the LoG threshold estimator

---

## Definition of done

- [ ] `int()` gone; `_integer_radius` is the only rounding in the function
- [ ] Tests, including a case where truncation and ceiling disagree by a whole pixel
- [ ] `make check` green; delta quantified against the table above, and the detection-quality
      block re-recorded with it
- [ ] ADR-0035; **B-064 filed**; `Backlog.md` (B-063 → done, and its overstated estimate
      corrected), `STATE.md`, `Progress.md`, `TASKS.md`, ADR index
- [ ] Commit: `M3-T24: the rough estimate stops truncating its own radius`

---

## Notes

This is the fourth `int()` this milestone has removed from a scientific path — after ADR-0024's
`min_size_pixel`, ADR-0020's un-rounded `disk()` radius and M3-T23's zero. The pattern is always
the same: a truncation written where a float was meant, invisible because it produces a plausible
integer.
