# ADR-0033 — A height that is not a number is not a measurement

- **Status:** Accepted
- **Date:** 2026-08-07
- **Affects:** `nanoscope/core/science/measurement/height.py` · **B-059** · M3-T22
- **Numerical impact:** **5 golden differences, all of them the new probe.** The fix itself moves
  nothing recorded, because **no phantom has an empty substrate** — which is exactly why the
  golden could not have caught the defect.

## Context

`measure_all_baseline` discards a particle whose height is not positive — "they are artefacts",
by the guard's own comment. It was written:

```python
if metrics["height_nm"] <= 0:
    continue
```

and **`nan <= 0` is `False`**, so the one value most obviously an artefact was the one value the
guard let through. Reproduced end to end on a constant map:

```
   particle_id  height_nm  baseline_nm baseline_source
0            0        NaN          NaN          global
1            1        NaN          NaN          global
```

Two rows in a table of measurements, and nothing said so.

### Where the `nan` comes from

1. `substrate_mask = z_above < threshold_otsu(z_above)`. A map with one value has no threshold
   that separates it, so Otsu returns the value and the mask is **empty** — 0 px of 4096.
2. `np.median` of an empty selection is `nan`, with a `RuntimeWarning` nobody sees.
3. A particle whose own ring is too small falls back to that global baseline, and its height is
   `nan`.
4. The guard above lets it into the table.

This is the same comparison **ADR-0018 already ruled on**, for the same reason, in the same
milestone — `not x > 0` and `x <= 0` differ precisely on `nan`, and `nan` is always the case that
matters. It was found while writing M3-T12's tests, and deferred twice (M3-T12, M3-T14) rather
than bundled into a schema change, which ADR-0010 requires.

## Decision

**The guard is `not metrics["height_nm"] > 0`, and the row is dropped.** A height that is not a
number is not a measurement. Keeping it would be shaped correctly and wrong: `height_nm` is
`float64` and `nan` is legal in it, so every consumer downstream — a mean, a histogram, a CSV
export — would have to know.

**And an empty substrate mask stops being silent.** The fix on its own turns two `nan` rows into
zero rows, which reads exactly like "there was nothing here" — the sentence that let this survive
a whole milestone. `global_baseline` is computed only when there is a substrate to compute it
from; otherwise it is `nan` and a warning names the cause and the consequence:

> the substrate mask is empty (Otsu threshold 3 on a map whose values do not separate), so there
> is no global baseline; any particle without a usable ring cannot be measured and is dropped

Same call ADR-0025 made when the `min_size_nm` filter cannot be applied: skipping is acceptable,
skipping silently is not.

**Not raised.** The image is valid — a flat scan is a real thing — and ADR-0018 settled that an
empty result is an answer rather than an error. Refusing the whole scan is ADR-0017's case, where
the *caller* asked for something the data cannot supply.

## Consequences

**Positive**

- The measurement table contains measurements. `df["height_nm"].mean()` cannot come back `nan`
  because of a row that should never have existed.
- A scan that cannot be measured says why, once, in a sentence naming the mechanism.
- The guard now means what its comment always claimed.

**Negative**

- A caller who was reading `nan` rows as "these particles exist but could not be measured" loses
  that signal. Nobody was: the rows carried `nan` in every numeric column, so they could not be
  distinguished from a failure of any other kind.

**Neutral**

- Every real measurement is unchanged. `not h > 0` and `h <= 0` agree on every number and differ
  only on `nan`.

## Found while testing: the empty substrate is all-or-nothing

There is no "partial success" on this route, and the code says so more strongly than expected.
`get_clean_ring` intersects the ring with the substrate mask, so **an empty substrate leaves every
particle without a ring**, every particle falls back to the baseline that is `nan`, and the whole
table goes. A scan where some rows survive a `nan` global baseline does not exist.

That is why the warning names the substrate rather than the dropped rows: the rows are never a
subset. It is pinned by a test rather than left as a comment.

## What is deliberately not in this commit

- **The two SAM2 producers.** Their baseline is the median of a ring required to have at least 5
  pixels, so there is no `nan` route to guard. A guard with no defect behind it is noise.
- **B-061** (a rough opening radius of 0) and **B-062** (recall 0.000 on `afm_sparse_low_snr`).
  Each moves numbers and needs a decision this one does not.
- Anything about Otsu's behaviour on a constant map. Returning the constant is correct; a map with
  one value has no split.

## The measured delta

**5 differences: `measure_all_baseline_empty_substrate: ADDED`, one per AFM phantom. Nothing
else.**

The fix changes no recorded value, and that is the finding rather than a reassurance: `not h > 0`
and `h <= 0` agree on every number, and **no phantom reaches the `nan` path at all**. The golden
had no way to see this defect, so the probe ships in the same commit — as M3-T07's
`negative_with_structure` and M3-T12's empty-blobs case did before it. Fifth time in this
milestone that closing a defect meant extending the harness that missed it.

What the probe records, on a constant map with blobs to measure:

```json
{"ok": true, "n_rows": 0, "columns": [13 names]}
```

Zero rows and the full schema — the two properties that had to hold together. Before the fix the
same probe would have recorded two rows of `NaN`; the guard is what changed, and ADR-0027's
declared columns are what did not.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Keep the row, with `nan` | Shaped right, wrong content. Pushes the check onto every consumer forever, and the consumers are notebooks and a GUI that does not exist yet | `nan` meant something specific here — it does not; it means "no baseline existed" |
| Raise when the substrate is empty | A flat scan is valid data with nothing measurable in it, which ADR-0018 already settled as an answer rather than an error | The caller had asked for something impossible (ADR-0017's case) |
| Substitute 0.0 for the missing baseline | The seventh substitute value this milestone would have had to delete, and the worst kind: it would make every height equal to the peak | The substrate were known to be at zero |
| Fix it inside `measure_height` instead | That function measures one particle and reports what it found; deciding whether a result is admissible belongs to the producer assembling the table | `measure_height` were private to this one caller |
| `math.isnan` check next to the `<=` | Two conditions where one suffices, and the two-condition form is what the next person deletes as redundant | The guard needed to distinguish `nan` from a negative height |

## Compliance

- `tests/unit/test_nan_height.py` — **10 tests**: the reproduction (no `NaN` reaches the table,
  and the table keeps its 13 columns while doing it), one assertion on the comparison itself
  because the next person to write this guard will reach for `<=` again, the warning firing on an
  empty substrate and **not** firing on an ordinary map, the unchanged behaviour on negative and
  zero heights, and the all-or-nothing property found while writing them.
  **Restoring `<= 0` turns 2 of the 10 red** — the two that assert the rows are gone. The other
  eight cover the warning and the behaviour this task must *not* change, and a mutant of the
  comparison correctly leaves them green.
- Golden: `measure_all_baseline_empty_substrate` on all five AFM phantoms.

## References

- `ADR-0018` — the ruling on `not x > 0` versus `x <= 0`, first applied to a `nan` maximum
- `ADR-0025` — absent is warned about, never silent
- `ADR-0027` — the empty table keeps its columns, which this commit must not undo
- **B-059**, closed by this task
