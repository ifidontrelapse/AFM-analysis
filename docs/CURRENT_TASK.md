# CURRENT TASK

**ID:** `M3-T22`
**Title:** A height that is not a number is not a measurement
**Milestone:** M3 — Numerical correctness, twentieth task
**Defect:** **B-059** (found while writing M3-T12's tests, deferred twice by ADR-0010) ·
**ADR:** **ADR-0033**
**Branch:** `sci/m3-numerical-correctness` (the consolidated branch — see the declared
deviation from PROJECT_RULES §7 in `STATE.md`)
**Status:** **done 2026-08-07.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

Of what is left in M3, it is the only item that is both unblocked and wrong: `M3-T16` waits on
**B6**, `M3-T19` is a `low` typing finding, `B-061` and `B-062` each need a decision this task
does not have to make. B-059 needs none — **the rule was decided in ADR-0018** and this is the
third site it applies to. It has been carried since 2026-08-06 because M3-T12 and M3-T14 both
refused to bundle a moved number into a schema change (ADR-0010), which was right, and the debt
comes due here.

## The defect, reproduced

```python
z = np.full((64, 64), 3.0)          # a constant map
measure_all_baseline(z, z, blobs)   # two blobs

   particle_id  height_nm  baseline_nm baseline_source
0            0        NaN          NaN          global
1            1        NaN          NaN          global
```

Two rows in the measurement table, both `NaN`, and **nothing said so**.

The mechanism, end to end:

1. `substrate_mask = z_above < threshold_otsu(z_above)`. On a constant map Otsu returns the
   constant, so the mask is **empty** — 0 px of 4096.
2. `global_baseline = float(np.median(z_flat[substrate_mask]))` — the median of nothing is `nan`,
   with a `RuntimeWarning` nobody sees.
3. Every particle whose ring is too small falls back to that baseline, so `height_nm` is `nan`.
4. `if metrics["height_nm"] <= 0: continue` — and **`nan <= 0` is `False`**, so the row that was
   supposed to be discarded is the one that survives.

Step 4 is the same comparison **ADR-0018 already ruled on**, for the same reason, in the same
milestone: `not x > 0` and `x <= 0` differ precisely on `nan`, and `nan` is always the case that
matters.

---

## The decisions this task has to make

### 1. What happens to the row

| | |
|---|---|
| **Drop it, as the guard always intended** ✅ | A height that is not a number is not a measurement. The guard exists to discard artefacts; a `nan` is the most artefactual value there is, and it survived only through a comparison bug |
| Keep the row with `nan` | `height_nm` is `float64` and `nan` is a legal value in it, so the table would be *shaped* correctly and *wrong*. Every consumer — a mean, a histogram, a CSV — would have to know |
| Raise | The image is valid and some particles may have measured perfectly through their own rings. Refusing the whole scan for the ones that could not is ADR-0017's case, and this is ADR-0018's |

### 2. Whether an empty substrate is allowed to be silent

**No.** Rows disappearing without a reason is how this defect stayed invisible: the fix alone
turns two `nan` rows into zero rows, which reads exactly like "no particles here". The empty
substrate mask is *the* diagnosable fact, and it gets a warning naming what happened — the same
call ADR-0025 made when the `min_size_nm` filter has to be skipped.

Partial success stays partial: a particle whose own ring gave it a baseline is unaffected and
keeps its row. Only the ones that fell back to a baseline that does not exist are dropped.

> **This paragraph was wrong, and the code said so.** `get_clean_ring` intersects the ring with
> the substrate mask, so there is no partial case at all — see *What it turned up*. The plan is
> left as written rather than quietly corrected, because the correction is the finding.

### 3. Whether the harness should have caught this

It could not: **no phantom has an empty substrate**, so the path has never been executed under the
golden. The probe is part of the fix, the way M3-T07's `negative_with_structure` and M3-T12's
empty-blobs case were — otherwise this commit makes the defect unreachable *and* unrecorded.

---

## Scope

**In scope**

1. `if not metrics["height_nm"] > 0` — the ADR-0018 comparison, at its third site
2. A warning when the substrate mask is empty, naming the consequence
3. A harness probe recording the constant-map case
4. Tests: the reproduction, the partial-success case, and that a legitimately negative height is
   still dropped exactly as before

**Out of scope**

- **B-062** (recall 0.000 on `afm_sparse_low_snr`) and **B-061** (a rough radius of 0) — each
  moves numbers and needs its own decision
- The two SAM2 producers. Their baseline comes from a ring that is required to have ≥ 5 px, so
  there is no `nan` route there; adding a guard would be a change with no defect behind it
- Anything about *why* Otsu returns the constant on a constant map. That is scikit-image's
  behaviour and it is not wrong — a map with one value has no threshold that separates it

---

## Expected blast radius, before measuring

- **Zero golden differences from the fix**, because no phantom reaches the path — and that is the
  finding, not a reassurance. The probe added in the same commit is what changes the file.
- No test should change meaning: the negative-height filter behaves identically on every number.

---

## Definition of done

- [x] `not metrics["height_nm"] > 0`, with the reason on the line
- [x] An empty substrate mask is warned about, once, naming what it costs
- [x] The harness records the constant-map probe
- [x] Tests — **10**: `nan` rows gone, legitimate rows kept, negative and zero heights still
      dropped. **The "partial case" could not be written — see below**
- [x] `make check` green — 425 tests; delta **5 differences, all of them the new probe**
- [x] ADR-0033; `STATE.md`, `Progress.md`, `TASKS.md`, `Backlog.md` (B-059 → done), ADR index
- [x] Commit: `M3-T22: a height that is not a number is not a measurement`

---

## What it turned up

**The planned "partial success" test does not exist as a case.** `get_clean_ring` intersects the
ring with the substrate mask, so an empty substrate leaves *every* particle without a ring, all of
them fall back to the `nan` baseline, and the whole table goes. There is no scan where some rows
survive it. That is why the warning names the substrate rather than the dropped rows — the rows
are never a subset — and it is pinned by a test instead of a comment.

**The fix moves nothing recorded, and that is the point.** No phantom has an empty substrate, so
the golden could not have caught this; the probe ships in the same commit. Fifth time in M3 that
closing a defect meant extending the harness that missed it.

---

## Notes

Third time in this milestone that `x <= 0` has been the wrong way to write it (ADR-0018,
ADR-0025's `not value > 0`, and here). If it happens a fourth time the rule belongs in
`PROJECT_RULES` §3 next to the unit conventions, not in three ADRs.
