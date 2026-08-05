# ADR-0020 — Opening radii are integers, rounded up

- **Status:** Accepted
- **Date:** 2026-08-05
- **Affects:** `nanoscope/core/science/preprocessing/substrate.py` · audit **D-10** · M3-T09 ·
  decision **B4**
- **Numerical impact:** **696 golden values move.** Every AFM phantom's opening radius grows by
  1–2 px, and every quantity downstream of the substrate moves with it. No particle count
  changes; measured heights move by at most **0.05 nm (0.37 %)**.

## Context

`skimage.morphology.disk(radius)` builds its element from `np.arange(-radius, radius + 1)`, so
the side length is even for a half-integer radius and there is no centre pixel:

| radius | `disk()` shape | centred |
|---|---|---|
| 8 | (17, 17) | yes |
| **8.5** | **(18, 18)** | **no** |
| 11.9 | (25, 25) | yes |

An uncentred structuring element biases the morphological opening by half a pixel, which shifts
the estimated substrate and therefore every height measured against it. Three sites fed `disk()`
a float:

```python
rough_radius   = max(radius_px * scale, min_size_pixel)     # float, from a function annotated -> int
opening_radius = max(int(typical_radius_px * 2.5), 5)       # int, but floored
opening_radius = manual_radius_px                           # whatever the caller passed
```

The third was left deliberately unrounded by **ADR-0014**, which recorded that rounding it would
pre-empt this decision.

Two facts narrow the choice more than the audit's table suggests. First, **any integer radius is
already centred** — `disk(r)` is `2r+1` on a side — so "round to the nearest odd radius" solves a
problem that rounding to an integer has already solved. Second, the three sites did three
different things, and the disagreement is itself the defect: one floored, one did nothing, one
was annotated as returning an integer and did not.

## Decision

**Every radius that reaches `disk()` is `ceil`-ed to an integer, and the rounding happens in the
one function they all pass through.**

```python
def _integer_radius(radius_px: float) -> int:
    return int(np.ceil(radius_px))

def get_substrate_map(z, radius_px: float):
    return morph_opening(z, disk(_integer_radius(radius_px))).astype(np.float32)
```

**Up, not down.** The opening radius must exceed the largest particle, or the disk fits inside a
particle and the "substrate" it recovers includes the particle's own top. Rounding down makes
that failure more likely; rounding up costs a slightly over-smoothed substrate, which is the
error the algorithm is designed to tolerate. Between a bias that corrupts heights silently and
one the method already absorbs, the choice is not symmetric.

**The guard lives in `get_substrate_map`, not at the call sites.** It is the funnel: every
caller, including any future one, goes through it, so one line fixes all three sites and cannot
be forgotten by the fourth.

**`build_substrate_map` also rounds the value it *reports*.** ADR-0014 made the manual branch
return the radius it actually uses; opening with 9 while reporting 8.5 would reinstate that lie
one field further along. `estimate_rough_radius` now returns the `int` its annotation always
claimed, on both of its exits — the flat-image fallback used to floor `width * 0.01` while the
main exit did not round at all.

## Consequences

**Positive**

- The structuring element is centred for every input. A property test asserts it across six
  radii, including the audit's 8.5.
- mypy: **18 → 15**. Three of the removed errors are the return-type lie and the `float` passed
  where an `int` was declared — the static shadow of this defect, present since M1-T04.
- One rule, one place. The three sites can no longer disagree.

**Negative**

- **696 recorded values move**, on all five AFM phantoms. This is the largest golden delta in M3
  so far, and it is not a bug being fixed in one number: the radius feeds the substrate, the
  substrate feeds `z_above`, and `z_above` feeds detection and every measurement. The delta is
  the propagation, not the defect's size.
- The opening is now marginally more aggressive on every scan, since radii only ever grow. On a
  scan where the previous radius was already generous, the substrate is smoother than before.

**Neutral**

- Particle counts are unchanged on every phantom — 24, 14, 59, 30, 0 before and after. The
  half-pixel bias moved heights, not detections.

## The measured delta

| phantom | opening radius | typical radius px | blobs (true) | mean height nm |
|---|---|---|---|---|
| `afm_flat_monodisperse` | 17 → **19** | 7.159 → 7.203 | 24 → 24 (24) | 16.1202 → 16.1194 |
| `afm_coarse_pixels` | 9 → **11** | 3.989 → 4.009 | 14 → 14 (14) | 17.8636 → 17.8664 |
| `afm_dense_overlapping` | 14 → **16** | 5.944 → 6.024 | 59 → 59 (70) | 13.3297 → **13.3791** |
| `afm_tilted_polydisperse` | 17 → **18** | 6.864 → 6.887 | 30 → 30 (30) | 16.1175 → 16.1030 |
| `afm_sparse_low_snr` | 7 → **8** | 2.877 → 2.877 | 0 → 0 (6) | — |

**Why +2 and not +1.** The radius is estimated twice. The rough radius is rounded up, which
changes the rough substrate, which changes the Otsu radii the final radius is derived from. The
second rounding then applies to a slightly different number. The cascade is expected — this is
the same two-stage estimate the code has always run, now with a consistent rounding rule at both
stages.

**The largest height move is 0.049 nm on `afm_dense_overlapping` (0.37 %)** — the phantom whose
particles touch, and therefore the one where a half-pixel of substrate bias has the most to bite
on. That number is the size of D-10 on data, and it is small, which is worth saying plainly: this
defect was worth fixing because it was silent and systematic, not because it was large.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Round to the nearest **odd** radius | Solves a problem integers already solve — `disk(r)` is `2r+1` for any integer `r`. It would also jump the radius by up to 2 px for no stated reason | `disk()` used a diameter rather than a radius |
| Floor | Systematically under-sizes the disk, and an opening radius smaller than a particle recovers a substrate that contains the particle | Over-smoothing were the dominant error |
| Round to nearest | Neither bound holds; half the scans get the failure mode floor has | — |
| Round at each of the three call sites | Three chances to forget, and a fourth caller inherits the bug. The funnel already exists | `get_substrate_map` were not the only route to `disk()` |
| Leave the manual radius exactly as the caller gave it | `disk()` still uncentres it — the caller's intent is honoured by opening with a centred element close to their request, not by passing a value that silently means something else | `disk()` accepted fractional radii meaningfully |

## Compliance

- `tests/unit/test_opening_radius.py` — 11 tests (6 parametrised): the element has a centre
  pixel for every radius including 8.5; rounding is up and leaves exact integers alone; both
  exits of `estimate_rough_radius` return `int`; a manual 8.5 is opened *and reported* as 9;
  and `get_substrate_map(z, 8.5)` is byte-identical to `get_substrate_map(z, 9)` and different
  from `get_substrate_map(z, 8)`. Restoring the floor turns 4 red — the centring property
  itself stays green, because floor also produces an integer; it is the *direction* the other
  four pin down.
- Golden: regenerated, 696 values, quantified above.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-10
- `ADR-0014` — the manual branch that deferred this decision, and the principle that the radius
  reported is the radius used
- Decision **B4**, answered by the operator 2026-08-05
