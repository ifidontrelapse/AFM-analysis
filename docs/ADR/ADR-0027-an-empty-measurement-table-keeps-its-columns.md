# ADR-0027 — An empty measurement table keeps its columns

- **Status:** Accepted
- **Date:** 2026-08-06
- **Affects:** `nanoscope/core/science/measurement/height.py` · audit **D-08** · M3-T12
- **Numerical impact:** **78 golden differences, all of them columns appearing where there
  were none.** No measured value moves. One of them is not the synthetic probe:
  `afm_sparse_low_snr` detects **0 blobs on its ordinary path**, so its real measurement
  table was zero-column too.

## Context

```python
df = pd.DataFrame(results)     # results == [] -> a DataFrame with zero columns
```

`measure_all_baseline` drops a particle for two ordinary reasons: its mask runs past the image
edge (`mask.sum() < 4`), and its measured height is not positive. When those take the last row —
or when detection found nothing to begin with — the function returned a table with **no columns
at all**. Every consumer reading a column by name then raised, and the audit reproduced it end to
end:

```python
>>> plot_pipeline_result(result_with_no_particles, z, scan)
KeyError: 'height_nm'
```

The failure is not in the measurement. It is that "no particles" and "no such column" are the
same object, so a caller cannot ask the first question without handling the second.

An empty result is an ordinary scientific outcome: a clean substrate, a threshold set high, a
region with nothing in it. The pipeline is expected to say so in a way a reader can consume.

## Decision

**The baseline schema is declared, and it is returned whether or not any row survived.**

```python
BASELINE_COLUMNS: dict[str, str] = {
    "particle_id": "int64", "x_px": "int64", "y_px": "int64", "sigma_px": "float64",
    "radius_nm": "float64", "method": "str", "height_nm": "float64", "mean_nm": "float64",
    "baseline_nm": "float64", "area_px": "int64", "ring_px": "int64", "baseline_source": "str",
}

def empty_baseline_table() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in BASELINE_COLUMNS.items()})
```

**Dtypes are part of the promise, not decoration.** `df["height_nm"].mean()` on an empty `str`
column is not the answer it is on an empty `float64` one, and a caller concatenating an empty
result with a populated one gets a different frame if the dtypes disagree.

**The declaration is checked against the populated path by a test**, not by inspection. A
declared schema is worth exactly as much as its agreement with what the code emits, and the
golden cannot catch a drift here: its empty case has no columns to compare against.

**`df.empty` stays the way to ask.** No sentinel, no `None`, no exception — the caller's question
is "were there any particles", and it keeps its natural spelling.

## Consequences

**Positive**

- D-08 is closed at the site the audit named. A consumer can read any column without first
  asking whether the table has one.
- The empty case is now recorded by the golden with content rather than with `[]`, so a future
  change to the schema shows up as drift.
- `pd.concat` of an empty and a populated result is the populated result, exactly.

**Negative**

- The schema is written in two places — the declaration and the row literal — and only a test
  keeps them together. The single-source alternative would be a typed record, which is
  **M3-T14**'s decision to make across all four producers, not this task's to pre-empt.
- One more public name in `core.science.measurement`. `empty_baseline_table` is exported because
  a caller assembling results across scans needs the same empty frame.

**Neutral**

- Nothing about how a height is measured changes; no populated row moves.
- The three other measurement producers still return zero-column frames when empty. That is not
  an oversight — see below.

## What is deliberately not in this commit

- **`run_sam2_from_blobs` and `run_sam2_from_boxes`.** They build each record with `if k in res`,
  so the column set varies **per row**, not just per call. Declaring a stable schema for them
  means deciding what that schema is, which is **D-16 / D-17 / M3-T14**. Doing it here would
  produce a schema M3-T14 has to undo.
- **`run_pipeline`'s detect-mode `pd.DataFrame()`.** The same zero-column value, but the right
  columns there depend on the modality — AFM heights or SEM/TEM geometry — and that is the
  question M3-T14 answers. Filed, not fixed.

## Found while testing, not fixed here

`measure_all_baseline` drops a row with `if metrics["height_nm"] <= 0`, and **`nan <= 0` is
`False`**, so a NaN height survives into the table. It is reachable on a constant map:
`substrate_mask` is empty, `np.median` of nothing is `nan`, and `global_baseline` carries it into
every height.

**ADR-0018 has already ruled on this exact comparison** — the guard must be `not height > 0`,
because that and `<= 0` differ precisely on `nan`. It is filed as **B-059** rather than fixed
here, because it moves a number and ADR-0010 keeps one defect to one commit.

## The measured delta

**78 differences: 72 keys added, 6 lengths changed, 0 values moved.**

Twelve column names appear, plus `columns: length 0 -> 12`, in six places: the
`measure_all_baseline_empty_blobs` probe on all five AFM phantoms, **and
`afm_sparse_low_snr`'s ordinary `measure_all_baseline` run**.

That sixth entry is the one worth reading. `afm_sparse_low_snr` detects **0 blobs** at the
harness's threshold, so its measurement table was the zero-column one on the normal path — not in
a probe written to provoke the defect, but in the run the phantom exists to represent. D-08 was
live on one of the five AFM phantoms, and the golden had been recording `columns: []` for it
since the baseline was taken.

No populated table changes: the four phantoms that measure particles keep every value and every
column, which is the evidence that the declared schema is the one the code already emitted.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Leave `pd.DataFrame([])`; make consumers guard with `if df.empty` | Pushes an invariant onto every reader, forever, and the readers are notebooks and a GUI that does not exist yet. It also does not help the reader who is *aggregating*: an empty frame with no columns poisons a `concat` | The library had one consumer |
| Return `None` when nothing survived | A second empty-ish value to check, and `None.empty` is an `AttributeError` one call later. ADR-0019: a second way to say "nothing" is a second thing to check | The return were already optional |
| Raise when nothing survived | An empty scan region is not an error. ADR-0017 raises when the *caller* asked the impossible; here the data simply contains nothing, which is ADR-0018's case | Measurement were only ever called on a known-populated image |
| Infer the columns from the row literal at import time | There is no row literal to inspect without running the loop; the keys come from two places, one of them another function's return | The record were a dataclass — which is M3-T14 |
| Declare the schema for all four producers now | Two of them do not have a stable schema to declare, which is a defect of its own (D-16/D-17). One task, one intent | M3-T14 had already run |

## Compliance

- `tests/unit/test_measurement_schema.py` — 7 tests in two classes. Empty: no blobs yields every
  column; a consumer reads `height_nm` without a `KeyError`; the dtypes match the declaration;
  and a run where *every* blob was rejected reads the same as one where none was detected.
  Populated: the emitted columns and dtypes are exactly the declaration — the drift guard — and
  `pd.concat` of the two is the populated frame. **Restoring `pd.DataFrame(results)` turns 3
  red.**
- Golden: `measure_all_baseline_empty_blobs.columns` moves from `[]` to the twelve names.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-08
- `ADR-0018` — zero particles is an answer, not an error; and the `not x > 0` rule B-059 needs
- **M3-T14** — one schema across the four producers, and the `bbox` contract
