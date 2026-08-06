# ADR-0024 — The minimum particle size is a physical size

- **Status:** Accepted
- **Date:** 2026-08-06
- **Affects:** `nanoscope/core/science/preprocessing/substrate.py` · audit **D-04** · M3-T02 ·
  decision **B2**
- **Numerical impact:** **47 golden differences** — 27 values changed, 15 keys added, 5 removed.
  All but one changed value is on `afm_sparse_low_snr`, where the size filter drops **75
  objects to 17**. The other four AFM phantoms are byte-identical; what moves on them is the
  error message's units.

## Context

`min_size_nm` is the one parameter in the preprocessing chain that a user states in physical
units. It was compared in pixels:

```python
min_size_pixel = int(min_size_nm / pixel_size_nm)      # three call sites
radii_px = radii_px[radii_px >= min_size_pixel]        # one comparison
```

Two things are wrong with the conversion, and only the second is obvious.

**It floors to zero.** The audit measured 120 of the operator's scans; this ADR re-read the
headers of all 628 in `data/`, and the ratio is the same:

| | audit, 120 scans | re-measured, 628 scans |
|---|---|---|
| `pixel_size_nm` min / **median** / max | 1.95 / **9.77** / 29.30 | 0.98 / **9.77** / 29.30 |
| scans where `int(5 / pixel_size_nm) == 0` | **108 (90 %)** | **568 (90 %)** |

A threshold of 0 admits every connected component that Otsu produced, single-pixel noise
included. Those radii are the input to `typical_radius_px`, which sets the opening radius for the
final substrate **and** the LoG sigma range. The corruption enters at the earliest numerical
stage and everything downstream inherits it, silently, on the majority of real data.

**And it means different things at different scales.** Even where the conversion does not floor
to zero, `int()` quantises the threshold to whole pixels: at 1.95 nm/px a 5 nm minimum becomes
2 px = 3.9 nm, and at 2.6 nm/px it becomes 1 px = 2.6 nm. The same stated minimum admits
different physical sizes on two scans of the same sample. A physical parameter that changes
meaning with the instrument's zoom is not a physical parameter.

The unit trail is the giveaway: `min_size_nm` is converted to pixels, compared against
`radii_px`, and then — three lines later, twice, identically — `radii_px` is converted back to
`radii_nm` for the result. The nanometre values the comparison wants already exist.

## Decision

**The filter compares nanometres with nanometres. There is no pixel conversion, and no `int()`.**

```python
radii_px = np.array([p.equivalent_diameter_area / 2 for p in props])
radii_nm = radii_px * pixel_size_nm

keep = radii_nm >= min_size_nm
radii_px, radii_nm = radii_px[keep], radii_nm[keep]
```

`estimate_radius_otsu` and `estimate_rough_radius` take `min_size_nm`, not `min_size_pixel`, so
the parameter keeps its unit from the caller all the way to the comparison and the three
conversion sites in `build_substrate_map` are gone rather than fixed.

**Nanometres, not "a floor of at least 1 px".** The floor was the other candidate answer to B2
and it is the one that looks safer: it guarantees the filter does something on every scan. It
was rejected because it re-introduces the same fault in a smaller font — at 29.3 nm/px, 1 px is
29.3 nm, so a "minimum of 5 nm" would silently discard everything under 29.3 nm on the coarsest
scans, which is most of a real particle-size distribution. The failure mode of the nanometre
comparison, by contrast, is that a sub-pixel threshold filters nothing on a coarse scan — which
is *correct*: if one pixel is 9.77 nm, no object smaller than 5 nm is resolvable, and there is
nothing to remove. **The threshold is right; it was the resolution that was never the question.**

**`estimate_rough_radius` still needs pixels**, because it is comparing against a radius derived
from a pixel area, so it converts — once, locally, and without `int()`:

```python
min_size_px = min_size_nm / pixel_size_nm
...
rough_radius = max(radius_px * scale, min_size_px)
```

The result then goes through `_integer_radius` (ADR-0020), which is where rounding belongs: at
the point where a radius reaches `disk()` and must be an integer, not at the point where a
physical minimum is stated.

**The error message speaks nanometres.** ADR-0017 made the empty-after-filter case name the
parameter, its value, and the largest object measured; the parameter is now physical, so all
three are (`min_size_nm=500 nm (the largest is 26.2 nm)`). This is a message we wrote, so
ADR-0022's comparator keeps checking it exactly.

**The duplicated `radii_nm` assignment goes with the change.** The audit's §Duplication entry —
the same line written twice in a row — was left alone by M3-T01, M3-T06 and M3-T09 on purpose,
because ADR-0010 keeps tidying out of numerical commits. It is fixed here because this change
*forces* it: the filter needs `radii_nm` before it runs, the assignment moves above the filter,
and the second copy has nowhere left to be. It is not tidying that came along for the ride.

## Consequences

**Positive**

- The noise filter is on for the first time on 90 % of the operator's scans, which is what D-04
  cost and the reason it was rated `critical`.
- `min_size_nm` means one thing at every pixel scale. Two scans of the same sample at different
  zoom now filter identically in nanometres.
- The unit suffix convention in PROJECT_RULES §3 is now true of this parameter end to end: no
  `_nm` value is compared against a `_px` value anywhere in the file.
- One fewer duplicated line, removed by the change rather than beside it.

**Negative**

- **The object count on a noisy scan falls by 77 %** (`afm_sparse_low_snr`, 75 → 17), and every
  quantity derived from the surviving radii moves with it — the radius distribution, the rough
  stage's Otsu threshold, and the LoG sigma range. A user comparing a result to one produced
  last week will see a different number of objects, and the new one is the defensible one.
- **The filter now bites, which means `estimate_radius_otsu` can now raise where it previously
  could not.** ADR-0017 predicted exactly this: it wrote the empty-after-filter error and then
  noted that most real scans never reach it *because D-04 floors the threshold to 0*. That
  protection is gone by design. A caller who asks for a minimum larger than anything in the
  image now gets the error instead of a silently unfiltered result — which is the trade ADR-0017
  already argued for, arriving on schedule.
- Small-particle studies that relied on the disabled filter will see fewer objects. There is no
  compatibility path and none is offered: the previous behaviour was not a policy, it was
  `int()`.

**Neutral**

- Nothing about *how* a radius is measured changes. `equivalent_diameter_area / 2` is untouched,
  and so is the two-stage rough → Otsu structure.
- `min_size_nm` remains a `build_substrate_map` argument with a default of 5. Promoting it to
  `PipelineConfig` is a configuration change, not a numerical one, and is not made here.
- **mypy is unchanged at 15**, and that is the point worth recording. M3-T09 and M3-T11 each
  removed errors that were their defect's static shadow; this one has none, because
  `int(float) -> int` is impeccably typed. A unit error is invisible to a type checker that
  cannot tell a nanometre from a pixel. The `_nm` / `_px` suffix convention in PROJECT_RULES §3
  is the only checker this class of defect has, and it is read by people.

## The measured delta

**47 differences: 27 values changed, 15 keys added, 5 removed.**

| phantom | nm/px | old threshold | new threshold | objects kept | typical radius px |
|---|---|---|---|---|---|
| `afm_flat_monodisperse` | 2.00 | 2 px | 2.5 px | 24 → 24 | 7.203 → 7.203 |
| `afm_coarse_pixels` | 9.77 | **0 px** | 0.512 px | 14 → 14 | 4.009 → 4.009 |
| `afm_dense_overlapping` | 2.00 | 2 px | 2.5 px | 51 → 51 | 6.024 → 6.024 |
| `afm_tilted_polydisperse` | 2.00 | 2 px | 2.5 px | 29 → 29 | 6.887 → 6.887 |
| `afm_sparse_low_snr` | 2.00 | 2 px | 2.5 px | **75 → 17** | 2.877 → **2.985** |

Everything else that moves is on `afm_sparse_low_snr` and follows from those 58 removals: the
radius distribution (`min` 2.03 → 2.52 px, `sum` 369 → 150, `std` 8.97 → 21.2), the rough
stage's Otsu threshold (1.168 → **1.459**, because the rough radius is derived from a floor that
is now 2.5 px rather than 2), and the LoG sigma range downstream (`max_sigma` 86.8 → **132.3**).
The final `opening_radius` is **8 on both sides**, so `substrate` and `z_above` are
byte-identical and **no measured height moves on any phantom**.

The five removed keys are `min_size_pixel_used`; the fifteen added are its replacement — the
physical threshold, its pixel equivalent, and the floored value the code used to compute, so the
golden keeps recording the arithmetic this ADR removes.

### The phantom built for D-04 does not move, and that is the finding

`afm_coarse_pixels` exists because at 9.77 nm/px `int(5 / 9.77)` is 0; the characterization
baseline records `min_size_pixel_used: 0` there as the defect's own fingerprint. Its numbers are
unchanged. **The smallest object a labelling can produce is one pixel, whose equivalent radius is
`sqrt(4/π)/2 = 0.564 px`** — 5.51 nm at that scale. A 5 nm minimum cannot remove it, floored or
not, so on that phantom the broken filter and the correct one agree.

Re-measuring the header of every scan in `data/` — 628 of them, five times the audit's sample —
splits the fleet into three regimes:

| pixel scale | scans | what `int()` did | what the fix changes at the 5 nm default |
|---|---|---|---|
| ≥ 8.86 nm/px | **365 (58 %)** | floored to 0 | nothing: one pixel is already over 5 nm |
| 5 – 8.86 nm/px | **203 (32 %)** | floored to 0 | **the filter starts working** — single-pixel noise is 3–5 nm here |
| ≤ 5 nm/px | 60 (10 %) | quantised down (≥ 1 px) | **the filter stops being lenient** — 2.5 px was truncated to 2 |

The audit's headline — *108 of 120 scans (90 %) get a threshold of zero* — reproduces exactly at
this sample size (**568 of 628, 90 %**). What the re-measurement adds is that the zero is only
*consequential* on the middle band, and that the finest 10 % were harmed by a mechanism the audit
did not name: not the floor to zero, but the truncation. `afm_sparse_low_snr` is in that band at
2 nm/px, where `int()` turned a 2.5 px threshold into 2 px and **58 of its 75 "objects" were
noise living in that half-pixel**. The other three 2 nm/px phantoms are clean, so nothing of
theirs sits between 2 and 2.5 px and nothing of theirs moves.

Which is the honest summary of D-04's size: the defect was real on 90 % of scans, it cost
nothing on 58 % of them, and where it cost something it cost **77 % of the object count**.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| `max(int(min_size_nm / pixel_size_nm), 1)` — a floor of one pixel | Keeps the unit confusion and changes the physical threshold with the zoom; at 29.3 nm/px it silently raises a 5 nm minimum to 29.3 nm | The parameter meant "ignore unresolvable objects" rather than "ignore small particles" |
| `round()` instead of `int()` | Halves the error and keeps all of the cause. `round(5 / 9.77) == 1` is still a pixel quantisation of a nanometre quantity | The comparison had to happen in pixels for some other reason |
| Raise when the conversion floors to zero | Turns the commonest real configuration into an error the operator cannot act on — their scan resolution is not a mistake | A zero threshold were unrepresentable rather than merely wrong |
| Convert `radii_px` to nm at the call site and keep the pixel signature | Moves the unit boundary outward without removing it, and leaves three call sites free to disagree again | `estimate_radius_otsu` had callers that genuinely work in pixels |
| Fix it silently as part of M3-T06, which touched the same filter | ADR-0010: one defect, one commit. M3-T06's delta would then have been unattributable, and B2 was still open | The two defects had a single cause |

## Compliance

- `tests/unit/test_substrate.py::TestTheMinimumSizeIsPhysical` — 5 tests over a fixture of three
  blocks with **exact** pixel areas (64, 16 and 1 px², so the equivalent radii are arithmetic
  rather than an artefact of a Gaussian's tail against Otsu): single-pixel noise is filtered at
  6 nm/px where `int()` floored to zero; a 2.5 px threshold is not truncated to 2 px at
  2 nm/px; nothing is removed at 9.77 nm/px **and that is correct**, because one pixel is
  already 5.5 nm there; the same stated minimum keeps the same physical sizes at 1.0, 2.0, 6.0
  and 9.77 nm/px; and the error message is in nanometres with no `px` left in it.
  **Restoring the `int()` turns 3 of the 5 red** — the two that stay green are the ones that
  document a regime where the two arithmetics agree, and the error message, which the
  restoration does not reach.
- The existing `TestOtsuSizing` cases move to `min_size_nm` and pin ADR-0017's behaviour
  unchanged; `test_logging.py` and `test_opening_radius.py` follow the signature.
- Golden: regenerated, quantified above.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-04 — the 108/120 measurement
- `docs/audit/characterization-baseline.md` §3.4 — `min_size_pixel_used: 0` on `afm_coarse_pixels`
- `ADR-0017` — the empty-after-filter error this change makes reachable
- `ADR-0020` — where rounding belongs: at `disk()`, not at the parameter
- `ADR-0010` — why this is its own commit
- Decision **B2**, answered by the operator 2026-08-05
