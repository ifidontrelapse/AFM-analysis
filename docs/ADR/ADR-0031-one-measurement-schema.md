# ADR-0031 — One measurement schema, and a `bbox` that means something

- **Status:** Accepted
- **Date:** 2026-08-07
- **Affects:** `nanoscope/core/science/measurement/schema.py`, `height.py`,
  `nanoscope/infrastructure/models/sam2.py`, `nanoscope/core/entities/detection.py`,
  `nanoscope/application/use_cases/pipeline.py` · audit **D-16**, **D-17** · M3-T14
- **Numerical impact:** **62 golden differences, every one of them a name, a dtype or a column
  that appeared — and 35 column digests unchanged, 0 changed.** The renamed column's digest is
  byte-identical to the one it replaced, which is the proof that a rename is what happened.

## Context

Four producers, four schemas:

| Producer | Columns |
|---|---|
| `measure_all_baseline` | `particle_id x_px y_px sigma_px radius_nm method height_nm mean_nm baseline_nm area_px ring_px baseline_source` |
| `run_sam2_from_blobs` (AFM) | `x_px y_px score height_nm baseline_nm peak_nm mask_area_px log_radius_nm` |
| `run_sam2_from_boxes` (AFM) | `x_px y_px sam_score height_nm baseline_nm peak_nm mask_area_px` |
| `run_sam2_*` (SEM/TEM) | `x_px y_px score/sam_score area_px area_nm2 radius_px radius_nm circularity aspect_ratio` |
| `run_pipeline`, detect mode | *(none — `pd.DataFrame()` has zero columns)* |

Reading them for this task found three faults where the audit named one.

**One quantity under two names.** `score` and `sam_score` are the same SAM2 number, from two
functions that were copy-pasted and then drifted — the audit spotted this one. `area_px` and
`mask_area_px` are the same count of pixels, and it did not.

**Two quantities under one name, which is worse.** `radius_nm` is the *detector's blob radius* in
`measure_all_baseline` and the *measured mask's* equivalent radius in the SEM/TEM SAM2 path. A
reader who concatenates those tables gets one column holding two different measurements, and
nothing in the data says so. The first fault makes a consumer write more code; this one makes it
compute the wrong number.

**Columns that vary per row.** Both SAM2 producers assemble a record with `if k in res`, so two
particles in one call can have different columns and the DataFrame is their union, with NaN
wherever a key happened to be absent — for reasons that were never written down.

**D-16** is the same defect one layer up: `bbox: tuple[int, int, int, int] = field(default_factory=tuple)`
promises four ints and produces zero.

**The audit's yardstick no longer exists.** Its table measures each producer against the
TypeScript `ParticleMeasurement` interface, and ADR-0012 deleted the frontend. The schema had to
come from what the science produces and what a consumer needs.

## Decision

### A core, plus blocks that are present in full or absent in full

```python
CORE_COLUMNS          particle_id, x_px, y_px, area_px, method      every row, every producer
DETECTOR_COLUMNS      sigma_px, detector_radius_nm                  a detection prompted it
HEIGHT_COLUMNS        height_nm, baseline_nm, peak_nm, mean_nm,     AFM: a z map existed
                      baseline_source, ring_px
GEOMETRY_COLUMNS      radius_px, radius_nm, area_nm2,               a real mask was measured
                      circularity, aspect_ratio
SEGMENTATION_COLUMNS  mask_score                                    a segmenter scored its mask
```

`method` names the producer — `baseline_circle`, `sam2_blobs`, `sam2_boxes` — and therefore says
which blocks to expect.

**Not one wide table with NaN where a producer cannot fill a column.** That would make
`df["height_nm"]` readable everywhere, at the cost of saying SEM/TEM *has* heights and they are
all missing. It has no heights: the modality does not produce one. Six ADRs in this milestone
have turned on the difference between absent and substituted, and a column of NaN is a
substitution with better manners.

**A block is all-or-nothing.** That is what replaces `if k in res`: a producer declares its
blocks once per call and builds every row from the same list, so two rows in one table cannot
disagree about what was measured. It costs two numbers that were previously omitted —
`measure_all_baseline` now reports `peak_nm` (it is `height_nm + baseline_nm`, the definition it
was already computed from) and the SAM2 AFM path reports `mean_nm`, `ring_px` and
`baseline_source`, all of which it had and dropped.

### One name per quantity, and a different name per quantity

| Was | Is | Why |
|---|---|---|
| `score`, `sam_score` | **`mask_score`** | Not plain `score`: since ADR-0028 this project has two, and `Detection.confidence` is the detector's. This one is SAM2's predicted IoU for a mask |
| `mask_area_px`, `area_px` | **`area_px`** | The same count of pixels |
| `radius_nm` (baseline), `log_radius_nm` | **`detector_radius_nm`** | Where we looked |
| `radius_nm` (SAM2 SEM/TEM) | **`radius_nm`** | What we found |

`x_px` and `y_px` become **float64** in the baseline table. Three of the four producers already
emitted a subpixel centre; the fourth rounded, and the rounded value is the one it keeps — the
mask was built there, so reporting the blob's unrounded centre would misplace the measurement.
Only the dtype moves.

### `bbox` is `None` when there is no box

`tuple[int, int, int, int] | None = None`. A LoG detection has no bounding box, and `()` is a
four-element promise broken silently — the sixth substitute value this milestone has deleted
(after ADR-0019, 0024, 0025, 0027, 0028). The `type: ignore` M2-T02 wrote on that line to expire
itself expired.

### Detect mode returns the empty table of the schema it would have filled

`blocks_for(modality)`: AFM gets the height block, SEM and TEM get the geometry block, both with
zero rows. That is the modality-dependent case ADR-0027 named and deliberately left open.

## Consequences

**Positive**

- A consumer can write one loop over the output of all four producers, reading the core columns
  without asking which producer it holds.
- Concatenating two tables can no longer merge two different measurements into one column.
- No row in a table can disagree with its neighbours about what was measured.
- `Detection.bbox` is either four numbers or absent, and mypy checks it rather than being told to
  look away.

**Negative**

- Every existing consumer of `score`, `sam_score`, `mask_area_px`, `log_radius_nm` or the
  baseline's `radius_nm` has to be updated. In-tree that is the notebooks, which are experiments
  by declaration (M1-T09); out of tree it is nobody, because there is no release.
- The schema is now a module that four producers import, so a careless addition there changes
  four tables. The test that keeps that honest is the drift guard: the declaration is asserted
  against what each **populated** path emits.
- `peak_nm` and the SAM2 AFM path's three added columns are new values in the golden.

**Neutral**

- No measurement changes. Every number that was computed is still computed the same way; what
  moved is which name it is written under.

## What is deliberately not in this commit

- **A typed record class instead of a dict per row.** ADR-0027 named it as the single-source
  alternative, and it remains the right end state: it is a rewrite of every producer's internals
  and belongs with the persistence layer in **M4**, where the schema has to reach SQLite anyway.
- **B-059** — `nan <= 0` in `measure_all_baseline` lets a NaN height into the table. Still filed,
  still its own commit (ADR-0010).
- Any change to how a height or a geometry is computed.

## The measured delta

**62 differences: 60 in the baseline table, 2 in the `Detection` defaults. No value moved.**

Per AFM phantom, twice over — once for the populated run and once for the
`measure_all_baseline_empty_blobs` probe:

| | |
|---|---|
| `col::detector_radius_nm` | ADDED |
| `col::radius_nm` | REMOVED |
| `col::peak_nm` | ADDED |
| `col::x_px.dtype`, `col::y_px.dtype` | `int64` → `float64` |
| `columns` | length 12 → 13 |

plus `contracts.default_detection_bbox: [] -> None` and `default_detection_bbox_len: 0 -> None`.

**The rename is provably a rename.** Comparing the two golden files column by column:

- `col::radius_nm`'s digest before **equals** `col::detector_radius_nm`'s digest after, on all
  five phantoms — same min, max, mean, std, sum, percentiles;
- `x_px` and `y_px` are identical in every statistic, with only `dtype` differing;
- **35 column digests unchanged, 0 changed.**

`peak_nm` is the one added number, and it satisfies `peak_nm == height_nm + baseline_nm` on every
phantom, which is the definition `height_nm` was computed from in the first place.

**The SAM2 producers contribute zero differences, and that is not evidence.** Inference is
outside the gate (PROJECT_RULES §6) and there are no weights on this machine or in CI, so the
golden cannot execute either of them. Their delta is zero *by construction*. The 31 tests, driven
by a stub predictor, are the whole safety net for that half of D-17 — said plainly rather than
left for a reader to infer from a green run.

**The harness needed the same fix the code did.** `capture_contracts` did `list(det.bbox)`, which
is a `TypeError` the moment `bbox` can be absent — the assumption D-16 describes, living in the
tool that was supposed to detect it. It now records `None` for both keys, and `_len` is kept
rather than deleted: `0` was the defect and `None` is the absence that replaced it, and a reader
comparing the two golden files should be able to see that.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| One superset table, NaN where a producer cannot fill a column | Says SEM/TEM has heights that are all missing. `df["height_nm"].mean()` would then be `nan` rather than an error, which is a worse failure: it is answerable | Every modality could in principle produce every column |
| Keep four schemas, document them | Is today plus documentation. The reason a consumer cannot write one loop | The producers had genuinely disjoint outputs |
| Keep `score` and let the reader disambiguate by producer | The reader is a notebook, a CSV export (M4-T11) and a GUI table. Each would have to carry the mapping, and the CSV would carry it nowhere | There were one consumer |
| Emit `radius_nm` from both, add a `radius_source` column | Puts the distinction in a value instead of a name, so a `groupby` or a mean silently mixes them anyway | The two radii were the same kind of measurement |
| Fix `bbox` to `(0, 0, 0, 0)` | A box at the origin with no area is a plausible-looking wrong answer; `None` is unmistakable | Consumers could not handle `None` |

## Compliance

- `tests/unit/test_one_schema.py` — **31 tests over five tables**: the baseline producer and both
  SAM2 producers in both modalities, driven by a `StubPredictor` that returns three candidate
  masks and their scores exactly as `SAM2ImagePredictor` does, so `masks_pred[np.argmax(scores)]`
  is exercised rather than bypassed. Each table's columns must equal the declaration (the drift
  guard ADR-0027 established, now applied to all five); the core must be in every one; **no row
  may hold a missing value in a block it claims**, which is the `if k in res` fault stated as a
  property. Then one name per quantity, including a test that the detector radius and the measured
  radius are *genuinely different numbers* — the stub segments a disk of radius 6 while the prompt
  says 8 nm at 2 nm/px, so a reader who let them share a column would be averaging two things.
  Then the empty table of each kind, `blocks_for`, concatenation, and `bbox`.
- `TestNoNumberMoved` asserts inside the suite what the golden asserts outside it: the columns are
  the declaration's, `peak_nm` is `height_nm + baseline_nm`, `x_px` is float64, and its values are
  the **rounded** centres the masks were built at — not the blob's subpixel centre, which would
  have moved the measurement while claiming to rename it.
- Golden: 62 declared differences, listed above.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-16, §D-17
- `ADR-0027` — the empty table that keeps its columns, and the two cases it left to this task
- `ADR-0028` — `Detection.confidence`, which is why the segmenter's number is not called `score`
- `ADR-0012` — deleted the TypeScript contract the audit measured these producers against
