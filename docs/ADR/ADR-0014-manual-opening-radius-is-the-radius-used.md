# ADR-0014 — `build_substrate_map` reports the opening radius it was given

- **Status:** Accepted
- **Date:** 2026-08-04
- **Affects:** `nanoscope/core/science/preprocessing/substrate.py` · audit **D-01** · M3-T01
- **Numerical impact:** the manual branch goes from *always raising* to *returning*. No
  automatic-path number moves.

## Context

`build_substrate_map` has two branches. The automatic one estimates an opening radius in
two stages and assigns `opening_radius`. The manual one — taken when the caller passes
`manual_radius_px` — computes the substrate with that radius but **never assigns
`opening_radius`**, and the shared `return` statement then reads it:

```python
if manual_radius_px is not None:
    substrate = get_substrate_map(z, manual_radius_px)
    ...
    # opening_radius is never bound here
else:
    ...
    opening_radius = max(int(sizes["typical_radius_px"] * 2.5), 5)
    ...
return substrate, z_above, opening_radius, sizes
```

Every call with an explicit radius raises `UnboundLocalError`. Not sometimes — **100% of
calls**, on every input, since the branch was written. The characterization baseline records
that exception for all five AFM phantoms, so the defect is in the golden file as current
behaviour.

This is audit **D-01**, rated critical, and it is the first numerical fix in the project.

## Decision

**On the manual branch, `opening_radius` is `manual_radius_px` — the value actually passed
to `get_substrate_map`.**

```python
opening_radius = manual_radius_px
substrate = get_substrate_map(z, opening_radius)
```

The returned radius is documented as "the radius finally used". On this branch that is the
caller's value, by definition: it is the argument the morphological opening received.

**No rounding, no coercion, no floor.** The automatic branch produces
`max(int(...), 5)` — an int with a floor — and it would be easy to make the manual branch
match. That would be a second decision, and it is not this one:

- Rounding a half-integer radius is **open decision B4 / M3-T09**. `disk(4.5)` produces an
  even-sized structuring element with no centre pixel, which shifts `z_result` by half a
  pixel. Choosing a rounding rule here would pre-empt an operator decision about physics.
- Applying the automatic branch's floor of 5 would silently override an explicit request.
  A caller who asks for 3 and gets 5 has been lied to; if 3 is wrong, that is a validation
  error (M3-T13), not a quiet substitution.

So the manual branch reports what it did, and the question of what it *should* accept stays
where it belongs.

## Consequences

**Positive**

- The manual-radius path exists for the first time. It is the path an operator uses when the
  automatic estimate is wrong, which is the case on dense or aggregated samples.
- `PreprocessingResult.opening_radius` becomes truthful on both branches rather than
  unreachable on one.
- The golden gains real coverage where it previously recorded only an exception.

**Negative**

- `opening_radius` is `int` on the automatic branch and whatever the caller passed on the
  manual one, so the field's runtime type now depends on the branch.
  `PreprocessingResult.opening_radius` is annotated `int`. This is honest but untidy, and it
  is exactly the inconsistency **M3-T09** resolves when the rounding rule is decided.
- The fix makes a previously dead code path live, so its own defects become reachable: it
  shares `int(min_size_nm / pixel_size_nm)` with the automatic branch, which is **D-04**
  (zero on any scan coarser than 5 nm/px, open decision **B2**).

**Neutral**

- No automatic-path number changes. Verified: the only golden movement is the five
  `build_substrate_map_manual` entries.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| `opening_radius = int(manual_radius_px)` | Silently discards a fractional radius the caller chose, and pre-empts B4. | M3-T09 decides radii are always integers. |
| `opening_radius = max(int(manual_radius_px), 5)` | Overrides an explicit request without saying so. A caller asking for 3 is either right or should get an error. | Never — validation belongs in M3-T13. |
| Re-derive the radius via Otsu on the manual branch | Makes `manual_radius_px` advisory rather than manual, which is the opposite of what the parameter means. | — |
| Raise `NotImplementedError` and leave the branch dead | Honest, but the branch is documented, exercised by the golden, and needed for samples where the automatic estimate fails. | The parameter were being removed. |

## Compliance

- `tests/unit/test_substrate.py` — 6 tests, including that a different radius produces a
  different substrate (so the fix is not cosmetic), that the automatic branch is untouched,
  and that both branches agree when given the same radius. Restoring the bug turns 5 red.
- The golden records the manual branch's returned arrays and radius, not just `ok: true` —
  otherwise fixing the defect would leave the branch **less** characterized than while it
  was broken.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-01
- `ADR-0008` (the golden is the contract) · `ADR-0010` (isolated numerical changes)
- `docs/STATE.md` B4 — the open rounding decision this ADR deliberately does not make
