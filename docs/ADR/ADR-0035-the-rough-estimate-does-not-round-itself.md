# ADR-0035 — The rough estimate does not round its own radius

- **Status:** Accepted
- **Date:** 2026-08-08
- **Affects:** `nanoscope/core/science/preprocessing/substrate.py` · **B-063** · M3-T24
- **Numerical impact:** **730 golden differences across four AFM phantoms.** Large in count,
  small in magnitude: the mean measured height moves by **≤ 0.09 %**, 320 of the differences are
  under 1 %, and `afm_dense_overlapping` **gains one detected particle** (59 → 60). Detection
  quality is unchanged in recall and mixed in the second decimal.

## Context

```python
radius_px = int(np.sqrt(median_area / np.pi))   # truncates, downward, silently
rough_radius = max(radius_px * scale, min_size_px)
return _integer_radius(rough_radius)            # the declared rounding, upward
```

Two roundings in three lines, in opposite directions, and only the second one is documented.

**The rule was already decided, three times over:**

- **ADR-0020** — every radius reaching `disk()` is `ceil`-ed, `_integer_radius` is the one funnel,
  and the direction is *up*: "a radius smaller than a particle recovers a substrate containing the
  particle".
- **ADR-0024** — deleted this exact `int()` pattern as D-04's mechanism.
- **The parameter's own docstring**, from the March 2026 commit that introduced both the
  truncation and `scale=1.7`: *"a multiplier so the disk is safely **larger** than a particle"*.
  The truncation makes it smaller. It works against the purpose of the line below it.

So this ADR does not decide a rule. It applies one at the site that was missed.

The truncation also loses up to a whole pixel **before** the `× 1.7`, so the error is amplified:
at an equivalent radius of 6.60 the disk is built from 6, and the estimate comes out 11 instead of
12. And it is how M3-T23's zero arose — `int(0.56) = 0`.

## Decision

**`radius_px` keeps its fractional part.** `_integer_radius` is the only rounding in the function,
which is what ADR-0020 says it is.

Rejected: rounding to nearest instead (still a second, undeclared rounding ahead of an amplifying
multiply), and documenting the truncation (documents a contradiction rather than removing it).

**Not decided here:** whether `1.7` and `2.5` are the right constants. They are undocumented
beyond "safely larger", they were chosen *with* the truncation in place — same commit, adjacent
lines — and M3-T15 can now measure a candidate, which turns it from a preference into a task.
**Filed as B-064.**

## Consequences

**Positive**

- One rounding, declared, in the direction ADR-0020 chose. A reader of `estimate_rough_radius` no
  longer has to notice that two lines disagree about which way a radius rounds.
- Two particles whose equivalent radii differ by 0.35 px no longer produce the same disk: the
  truncation mapped every radius in `[3, 4)` to 3.
- The rough opening is now at least as large as the estimate implies, never smaller — which is the
  safety margin the `scale` parameter exists to provide.

**Negative**

- **730 recorded values move.** This is a bigger change than the task planned for; see below.
- `afm_dense_overlapping` detects one more particle. That is a *different answer*, not a more
  precise one, and nothing here says it is the right one — the evaluation harness reports the same
  recall either way.
- The constants stay unjustified, and are now applied to a slightly different input.

**Neutral**

- `afm_sparse_low_snr` does not move at all: its estimate is the fallback M3-T23 installed.

## The measured delta

**730 differences, four phantoms.** By magnitude, which is the number that matters:

| Relative change | Count |
|---|---:|
| under 0.1 % | 26 |
| 0.1 – 1 % | 294 |
| 1 – 5 % | 301 |
| 5 – 20 % | 88 |
| over 20 % | 21 |

The 21 over 20 % are baseline percentiles near zero, where a small absolute move is a large
relative one — `baseline_nm`'s 90th percentile goes 0.059 → 0.143 nm.

### What actually changed, in the units of the science

| Phantom | rough | final | mean height | detected |
|---|---|---|---|---|
| `afm_flat_monodisperse` | 14 → **15** | 19 → 19 | unchanged | 24 → 24 |
| `afm_tilted_polydisperse` | 12 → **14** | 18 → **19** | 16.103 → 16.089 nm (**0.09 %**) | 30 → 30 |
| `afm_dense_overlapping` | 11 → **12** | 16 → 16 | 13.379 → 13.374 nm (**0.04 %**) | **59 → 60** |
| `afm_coarse_pixels` | 7 → **9** | 11 → 11 | unchanged | 14 → 14 |
| `afm_sparse_low_snr` | 3 → 3 | 8 → 8 | unchanged | 0 → 0 |

The mean height moves in the third decimal. The 90th percentile of `afm_tilted_polydisperse`'s
heights moves 2.5 % (23.66 → 23.07 nm), which is the largest scientifically meaningful move here.

### Why the task's own estimate was 70× too small, and what that teaches

The plan predicted "one phantom's substrate and heights, one phantom's row count" — measured by
comparing the substrate arrays and the height columns directly. Both of those predictions held.
What the simulation did not model is that **`sizes` travels onward**:

```
rough radius → Otsu on the roughly-opened map → sizes
                                                  ├→ final radius   (absorbed: moves on 1 of 5)
                                                  └→ estimate_log_params → sigma range → every blob
```

`estimate_log_params` derives `min_sigma` and `max_sigma` from `sizes["radii_px"]`, so a rough
radius that moves shifts the sigma range the LoG detector searches — on every phantom, whether or
not the substrate moved. That is 379 of the 730 differences (`log_detection`) and most of the 224
under `baseline_measurement`.

**`afm_dense_overlapping` is the clean demonstration: its substrate is byte-identical and it still
detects one more particle**, purely because the sigma range shifted.

The lesson is about the two-stage design rather than about this defect: the second stage is robust
to the first (the final radius moved once in five), but the *diagnostics* the first stage produces
are not, and they are wired into the detector.

### Detection quality — the first time this question has been answerable

Recall is **unchanged on every phantom**. On `afm_tilted_polydisperse`, the only one whose
substrate moved:

| | before | after |
|---|---:|---:|
| mean radius error | 0.765 px | **0.718 px** |
| mean *signed* radius error | −0.704 px | **−0.669 px** |
| mean localisation error | 0.6137 px | 0.6156 px |
| median localisation error | 0.4802 px | 0.4770 px |

Radius error improves by 6 %, localisation degrades by 0.3 % in the mean and improves in the
median. **That is a wash, and it is reported as one.** This task is not an improvement to
detection; it is the removal of an undeclared rounding that contradicted an accepted ADR, and the
harness from M3-T15 is what makes that sentence checkable instead of assumed.

The signed radius error moving toward zero is consistent with the mechanism — a larger opening
leaves slightly more of each particle above the substrate — but 6 % on one phantom is not
evidence, and no claim is made.

## Compliance

- `tests/unit/test_rough_radius_rounding.py` — **18 tests**. The core assertion is the contract
  computed from the image the function was given: `ceil(sqrt(median_area / π) × scale)`, once. Then
  that the answer is strictly larger than truncation's on the radii where the two differ; that
  **where truncation happened to agree it still does** — at an equivalent radius of 3.432 both give
  6, which is stated rather than glossed, because pretending the defect showed everywhere would
  overstate it; a constructed collision pair (equivalent radii 3.432 and 3.785, both truncating to
  3) that now yields 6 and 7; and the three behaviours that must not move — the `min_size_nm`
  floor, M3-T23's sub-pixel fallback, and the empty-image fallback. **Restoring `int()` turns 11 of
  the 18 red.**
- Golden: 730 differences, tabulated above, including the `detection_quality` block.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Round to nearest | Still a second, undeclared rounding, still ahead of the multiply that amplifies it | `_integer_radius` did not exist |
| Keep it, document it | Writes down a contradiction with ADR-0020 instead of removing it, and leaves a downward rounding inside an upward safety margin | The truncation had a stated purpose — it has none |
| Fix it inside M3-T23 | It is the proximate cause of that task's zero, and bundling would have made a 730-value delta unattributable against an 11-value one (ADR-0010) | The two had the same blast radius |
| Change `1.7` to compensate, keeping the truncation | Two wrongs tuned against each other, and the compensation would be wrong for every image whose median area sits at a different place in the truncation step | The constant had been derived rather than chosen |
| Leave it — the deltas are small | The magnitudes are small; the *rule* is not optional. And "small" was not knowable before it was measured, which is the argument for measuring rather than for skipping | The project had no golden |

## References

- `ADR-0020` — the one rounding, and its direction; **B-064** for the constants it multiplies
- `ADR-0024` — the same `int()` pattern, deleted as D-04's mechanism
- `ADR-0032` / M3-T15 — the evaluation harness that makes the quality claim checkable
- **B-063**, closed here; **B-064**, filed here
