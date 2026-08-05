# ADR-0017 — Otsu sizing fails loudly, and counts what it kept

- **Status:** Accepted
- **Date:** 2026-08-05
- **Affects:** `nanoscope/core/science/preprocessing/substrate.py` · audit **D-05**, **D-06** ·
  M3-T06
- **Numerical impact:** `n_objects` changes wherever the size filter removes anything. On the
  phantoms that is a minority of cases today, for a reason that is itself a defect (D-04).

## Context

`estimate_radius_otsu` thresholds, labels, measures every object's equivalent radius, and
then drops the ones below `min_size_pixel`. Two things went wrong after that filter.

**D-05 — the empty case returns `nan`.** The guard covers `len(props) == 0` *before*
filtering. If the filter removes everything, `np.median([])` is `nan` with a `RuntimeWarning`
and nothing else:

```python
>>> estimate_radius_otsu(z, 1.0, min_size_pixel=500)
{'typical_radius_px': nan, 'radii_px': array([], dtype=float64), 'n_objects': 4, ...}
```

The `nan` becomes the LoG sigma range and surfaces two calls later as
`ValueError: zero-size array to reduction operation minimum which has no identity` — a
message that names neither the parameter that caused it nor the stage that produced it.

**D-06 — `n_objects` is the pre-filter count.** It returned `len(props)` while `radii_px`
had already been filtered. The audit measured 4 reported against 2 retained; on the noisiest
phantom the current baseline reports **1023**.

## Decision

**Raise when the filter empties the set, and report the count of what survived.**

```python
n_found, largest = len(radii_px), float(radii_px.max())
radii_px = radii_px[radii_px >= min_size_pixel]

if radii_px.size == 0:
    raise ValueError(
        f"Otsu found {n_found} objects, none with a radius of at least "
        f"min_size_pixel={min_size_pixel} px (the largest is {largest:.3g} px). "
        "Lower the minimum size, or check the preprocessing and the image quality."
    )
...
"n_objects": len(radii_px),
```

**The message names the parameter, its value, and the largest object measured.** PROJECT_RULES
§3 requires the first two. The third is here because without it the two failures a user can
actually be in — "this image has no particles" and "your minimum size is an order of magnitude
too large" — produce identical text, and the caller has no way to tell which. `7.53 px`
against `min_size_pixel=500` diagnoses itself.

**`ValueError`, not a project exception type.** The typed error taxonomy is **M3-T13**; it
will re-home this raise along with every other one. Inventing half a taxonomy in a numerical
commit would leave two conventions in the tree and pre-empt that task's design.

**`n_objects` means "objects kept", and is therefore `len(radii_px)`.** The alternative —
keeping the pre-filter number and adding a second key — was rejected: `PROJECT_CONTEXT.md`
documents the field as "object count", every caller already reads it as the number of
particles, and a field that means "objects Otsu saw before we discarded most of them" has no
consumer. If the pre-filter count is ever wanted, it is diagnostics and belongs in a log line,
not in a result dict two layers of the pipeline pass around.

## Consequences

**Positive**

- A failure now names its cause at the point it happens instead of two calls downstream.
- `n_objects` is finally the number the notebooks and `PROJECT_CONTEXT.md` always claimed it
  was. Both notebooks read it directly.
- `nan` no longer propagates into the LoG sigma range, which was the actual failure path.

**Negative**

- **A previously silent path is now an error, and it is reachable from
  `build_substrate_map`.** Any caller that passes an explicit `min_size_nm` large relative to
  its features gets an exception where it used to get a `nan` result it probably never
  inspected. That is the intent, but it is a behaviour change on a live path, not only in
  degenerate tests — and one of this project's own tests was relying on it (below).
- The error text enters the golden, which compares exception messages. That is **B-058**'s
  fragility, but for our own text rather than CPython's: it moves only when we move it.

**Neutral**

- **Most real scans never reach the new error**, because **D-04** floors `min_size_pixel` to
  0 on any scan coarser than 5 nm/px — 90% of the operator's data — so the filter removes
  nothing there. When **B2 / M3-T02** decides D-04's semantics, this error becomes reachable
  on real data, and that is the point at which its wording earns its keep.

## What the tests found

`test_a_different_radius_produces_a_different_substrate`, written for M3-T01 four commits
ago, **started failing** on this change. Its fixture has four 4.7 px particles and it called
`build_substrate_map(z, 1.0, 5, ...)` — a 5 px floor at 1 nm/px, which filters all four away.
The test was passing while the sizing silently returned `nan` for every radius it reported,
because it only compared `z_above` arrays and never looked at `sizes`.

That is D-05's blast radius in miniature: the defect is invisible precisely because the `nan`
sits in a field nobody checks until something far away divides by it. The test now passes
`min_size_nm=1` and says why.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Return an empty result (`typical_radius_px = None`, `radii_px = []`) instead of raising | Pushes the decision to every caller, and today's callers use the radius unconditionally — it would become a `None` propagating exactly as the `nan` did | The result type were an explicit `Optional` the callers must unpack |
| Fall back to the unfiltered median when the filter empties the set | Silently ignores the caller's stated minimum size, which is the same class of lie ADR-0014 refused for the opening radius | — |
| Keep `n_objects` pre-filter and add `n_objects_kept` | Two counts, one of which is a trap for whoever reads the shorter name. The existing name is documented as the particle count | The pre-filter count had a consumer |
| Define a typed `SizingError` here | That is M3-T13, wholesale. Half a taxonomy is worse than none | M3-T13 were already done |
| Fix D-04's flooring in the same commit | Open operator decision **B2**, and a separate defect (ADR-0010) | B2 were answered |

## Compliance

- `tests/unit/test_substrate.py::TestOtsuSizing` — 4 tests: the empty-after-filter raise, the
  message naming parameter/value/largest object, `n_objects` matching `len(radii_px)` on the
  audit's exact 4-objects-2-retained case, and the no-op case where nothing is filtered.
  Restoring the old behaviour turns **3 of the 4 red**; the fourth passes either way by
  design, because it is what guarantees the unfiltered phantoms do not move.
- The harness records `estimate_radius_otsu_all_filtered` — D-05's own reproduction, with
  `min_size_pixel=500` — so the golden holds the error instead of holding nothing.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-05, §D-06
- `ADR-0010` (one defect, one commit) · `ADR-0014` (the precedent for refusing to silently
  substitute a caller's value)
- **B2 / M3-T02** — D-04, the reason the new error is mostly unreachable today
- **M3-T13** — the typed error taxonomy that will re-home this `ValueError`
