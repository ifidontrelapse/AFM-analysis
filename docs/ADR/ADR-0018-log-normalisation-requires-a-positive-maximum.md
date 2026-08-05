# ADR-0018 — LoG normalisation requires a positive maximum

- **Status:** Accepted
- **Date:** 2026-08-05
- **Affects:** `nanoscope/core/science/detection/log.py` · audit **D-11** · M3-T07
- **Numerical impact:** one recorded number moves — an adaptive threshold of **2.4997**
  becomes **0.05**. Everything with a positive maximum, which is every phantom and every
  normal scan, is byte-identical.

## Context

LoG detection normalises before filtering, because `blob_log` responses scale with the input:

```python
z_norm = z_above / z_above.max()      # two call sites: the adaptive threshold, and detect_particles
```

Two inputs break it, and neither raises:

- **`max() == 0`** — a flat map. `0/0` is `nan` for every pixel, `blob_log` finds nothing, and
  `detect_particles` logs *"no particles found; try lowering the threshold"*. The operator is
  sent to tune a knob that cannot help, for a failure that is not about the threshold.
- **`max() < 0`** — a map that is negative everywhere. Division by a negative number **flips
  the topography**: the substrate ends up brighter than the peaks. Measured on a map of four
  caps sitting at −10 nm with peaks at −4 nm, `estimate_log_threshold_adaptive` returned
  **2.4997** — a threshold on a `[0, 1]`-normalised image, so nothing can ever exceed it.

`build_substrate_map` guarantees `z_above >= 0`, so the negative case is unreachable *through
that path*. `LogDetector.detect` is also called directly on raw SEM/TEM images (that is
**D-12**), where no such guarantee exists.

There is a third site, `estimate_log_threshold`, which has had the guard since it was
written: `3.0 * noise_std / z_max if z_max > 0 else 0.05`. So the module already knew the
answer in one of three places.

## Decision

**A non-positive or non-finite maximum ends the computation early, at the point it is
detected, with a message that names the maximum.**

```python
z_max = float(z_above.max())
if not z_max > 0:
    logger.warning("no positive signal above the substrate (max = %.3g); ...", z_max)
    return DEFAULT_THRESHOLD          # in estimate_log_threshold_adaptive
    return np.empty((0, 4))           # in detect_particles
```

**`not z_max > 0`, not `z_max <= 0`.** The two differ on `nan`, and `nan` is the case that
matters: `nan <= 0` is `False`, so the arithmetic comparison would let a `nan` maximum
through and the division would spread it across every pixel. The awkward-looking negation is
the whole point, and it is commented as such.

**Zero particles is the answer, not an error.** This is the opposite call from ADR-0017,
which made an empty result raise, and the difference is who is wrong. There, the *caller*
asked for a filter no object could pass — a request that cannot be satisfied. Here, the
*data* has no signal above the substrate, and "no particles above the substrate" is a true,
useful answer about a legitimate input: an empty region of a scan. An error would force every
caller to distinguish "flat" from "broken" in a `try`.

**The guard sits after `estimate_log_params(sizes)`.** A caller who passes an unusable `sizes`
dict should hear about that, not about the image. Input validation first, then the early
return.

**`DEFAULT_THRESHOLD = 0.05` is now named.** The value already appeared three times in this
module, twice as a bare literal. ADR-0018 does not change it — 0.05 is what the module has
always used when a threshold cannot be derived — it gives it a name and one more call site.

## Consequences

**Positive**

- A `nan` image is never constructed, so nothing downstream inherits it.
- The diagnostic names the cause: `no positive signal above the substrate (max = -4)` instead
  of a threshold suggestion.
- The adaptive threshold is now always in `(0, 1]`, which is the interval it is compared
  against. A test asserts that across five degenerate shapes.

**Negative**

- A map that is negative everywhere returns zero particles where it previously returned
  whatever `blob_log` made of an inverted image. Today that was also nothing, but it was
  nothing *by accident*; if a future sign convention makes such maps meaningful, this guard
  is the line to revisit, not the division.
- The golden now stores our own warning-adjacent behaviour in one more place (the recorded
  threshold), which is one more thing a deliberate change must update.

**Neutral**

- Every phantom, and every scan through `build_substrate_map`, has a positive maximum. The
  working path is untouched, and the golden proves it: only `negative_with_structure` moves.

## What it took to see it at all

The characterization harness recorded `detect_particles` on ten degenerate inputs, and
**D-11 was invisible in every one of them** — the function returned an empty `(0, 4)` array
before and after, because a `nan` image and a correctly-refused image both yield no blobs.
Two changes to the harness, in this commit:

1. **A `negative_with_structure` degenerate input.** The existing `all_negative` is a
   *constant* −5, and dividing a constant by its own maximum gives a constant, which hides the
   flip. Structure is what makes the inversion observable.
2. **Scalars are recorded instead of being written down as the string `"non-array"`.** That
   line is why a threshold of 2.4997 sat in the harness's output, unrecorded, since Phase 0.

Both are the M3-T01 principle again: a fix that leaves its own path uncharacterized is not
finished.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Raise, as ADR-0017 does for the empty filter | There the caller's request was impossible; here the data is simply flat, and that is a legitimate scan region with a true answer | The function's contract were "find at least one particle" |
| Normalise by `ptp()` (max − min) instead | Fixes the flat case only by accident (`ptp == 0` too) and changes the normalisation of **every** image, which moves every recorded detection. That is a different decision needing its own ADR and evidence | Detection quality on offset maps were shown to suffer |
| Normalise by `abs(max)` | Keeps a negative map "working" by inverting its meaning silently — the worst option, because it produces plausible numbers from a sign error | — |
| Guard only `detect_particles` | The threshold estimator is the site that produced the 2.4997, and it is public. Half a fix | — |
| Clip negatives to zero first | Silently rewrites the caller's data, and a fully negative map would become a flat one — the same failure with a longer path | — |

## Compliance

- `tests/unit/test_log_detection.py` — 11 tests (5 of them parametrised): the threshold stays
  in `(0, 1]` on zeros, a constant negative, a negative-with-structure, a positive map and a
  `nan`-bearing map; the flat map returns no particles **and** logs the real reason rather than
  "try lowering the threshold"; a `nan` maximum is caught; `sizes` is still validated first; a
  normal map still finds its four caps. Restoring the raw division turns 3 red.
- Golden: `estimate_log_threshold_adaptive` recorded for all 11 degenerate inputs;
  `negative_with_structure` records **0.05** where the old code produced **2.4997**.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-11, §D-12 (why the negative case is reachable)
- `ADR-0017` — the empty case that *does* raise, and why this one does not
- `ADR-0010` (one defect, one commit, one ADR)
