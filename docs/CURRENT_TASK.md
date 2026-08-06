# CURRENT TASK

**ID:** `M3-T12`
**Title:** An empty measurement table still has its columns
**Milestone:** M3 — Numerical correctness, eleventh task
**Defect:** **D-08** (high) · **ADR:** **ADR-0027**
**Branch:** `sci/empty-measurements-keep-their-schema` (stacked on `sci/spm-header-without-scan-size`)
**Status:** **done 2026-08-06.** Rewritten for the next task at the start of the next
session; the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

It is the last unblocked `high` defect. Every other one in M3 is `medium`.

```python
df = pd.DataFrame(results)     # results == [] -> a DataFrame with zero columns
```

`measure_all_baseline` drops a particle when its mask runs past the image edge (`mask.sum() < 4`)
and when its height comes out non-positive. Both are ordinary outcomes, and when they take the
last row the function returns a table with **no columns at all**, so every consumer that reads by
name raises `KeyError` instead of seeing an empty column:

```python
>>> plot_pipeline_result(result_with_no_particles, z, scan)
KeyError: 'height_nm'
```

The harness already records the reproduction — `measure_all_baseline_empty_blobs` — with
`columns: []`. That entry is the delta this task moves.

---

## Scope

**In scope**

1. The baseline measurement schema is **declared**: twelve columns, each with a dtype
2. `measure_all_baseline` returns it whether or not any particle survived — same columns, same
   dtypes, zero rows
3. A test that the declared schema and the schema of a **non-empty** result are the same thing,
   so the two paths cannot drift
4. **ADR-0027**

**Out of scope**

- **The other three producers' schemas.** `run_sam2_from_blobs` and `run_sam2_from_boxes` build
  each record with `if k in res`, so their column set varies *per row* — that is **D-16/D-17**,
  and unifying it is **M3-T14**. Declaring a stable schema for a producer whose schema is not yet
  decided would have to be undone by that task
- **`run_pipeline`'s detect-mode `pd.DataFrame()`.** Same zero-column defect, but which schema is
  correct there depends on the modality (AFM heights vs SEM/TEM geometry), and that is exactly
  what M3-T14 decides. Filed, not fixed
- Column *renaming* or reordering. The declared schema is the one the code already produces,
  written down — this task changes the empty case, not the populated one

---

## The decision

| | |
|---|---|
| Leave `pd.DataFrame([])` and make consumers guard | Pushes an invariant onto every reader, and the readers are notebooks and a GUI that does not exist yet |
| Return `None` when nothing survived | A second empty-ish value to check, and `None.empty` is an `AttributeError` one call later |
| **Declare the schema and always return it** ✅ | One shape for both paths; `df.empty` stays the way to ask, and every column is readable |

Dtypes are part of the promise, not decoration: an empty frame with `object` columns answers
`df["height_nm"].mean()` differently from a `float64` one.

---

## Definition of done

- [x] Zero surviving particles gives twelve columns, correct dtypes, zero rows
- [x] The populated path produces exactly the declared columns — proven, and it **caught a
      mistake**: pandas 3 infers `str`, not `object`, for the two text columns
- [x] `make check` green — 225 tests; delta: **78 differences, 0 values moved**
- [x] ADR-0027; `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index, Backlog
- [x] Commit: `M3-T12: an empty measurement table still has its columns`

---

## What it turned up

**The defect was live on a real phantom, not only in the probe.** `afm_sparse_low_snr` detects
**0 blobs** on its ordinary path, so `measure_all_baseline` — not
`measure_all_baseline_empty_blobs` — returned the zero-column table, and the golden had been
recording `columns: []` for it since M0. Six blocks moved, five of them the synthetic probe and
one of them the real run.

**A NaN height passes the non-positive filter** (`nan <= 0` is `False`) and reaches the table,
reachable on a constant map through an empty substrate mask. ADR-0018 already ruled on this exact
comparison. Filed as **B-059** rather than fixed here — ADR-0010, one defect per commit.

---

## Notes

`measure_all_baseline` is the only one of the four producers whose record shape is fixed today —
it builds every row from the same dict literal plus `measure_height`'s six keys. That is why it
can have a declared schema now and the SAM2 pair cannot.
