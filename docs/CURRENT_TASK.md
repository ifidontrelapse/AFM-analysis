# CURRENT TASK

**ID:** `M3-T14`
**Title:** One measurement schema across the four producers, and a `bbox` that means something
**Milestone:** M3 — Numerical correctness, eighteenth task
**Defects:** **D-16**, **D-17** (medium) · **ADR:** **ADR-0031**
**Branch:** `sci/m3-numerical-correctness` (the consolidated branch — see the declared
deviation from PROJECT_RULES §7 in `STATE.md`)
**Status:** **done 2026-08-07.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

It is the last `medium` defect, and the last one M3-T12 named when it declared the baseline
schema and deliberately left the other three producers alone: *"the two SAM2 producers vary their
columns per row, and detect mode's schema is modality-dependent — that is M3-T14's decision."*

## The defect

**D-17 — four producers, four schemas.** What each emits today:

| Producer | Columns |
|---|---|
| `measure_all_baseline` | `particle_id x_px y_px sigma_px radius_nm method height_nm mean_nm baseline_nm area_px ring_px baseline_source` |
| `run_sam2_from_blobs` (AFM) | `x_px y_px score height_nm baseline_nm peak_nm mask_area_px log_radius_nm` |
| `run_sam2_from_boxes` (AFM) | `x_px y_px sam_score height_nm baseline_nm peak_nm mask_area_px` |
| `run_sam2_*` (SEM/TEM) | `x_px y_px score/sam_score area_px area_nm2 radius_px radius_nm circularity aspect_ratio` |
| `run_pipeline`, detect mode | *(none — `pd.DataFrame()`)* |

Three separate problems live in that table, and only the first is the one the audit named:

1. **One quantity, two names.** `score` and `sam_score` are the same SAM2 number, emitted by two
   functions that were copy-pasted and then drifted. So are `area_px` and `mask_area_px`.
2. **Two quantities, one name.** `radius_nm` is the *detector's blob radius* in
   `measure_all_baseline` and the *measured mask's* equivalent radius in the SEM/TEM SAM2 path.
   That is worse than the first problem: a reader who concatenates two tables gets one column
   holding two different measurements, and nothing says so.
3. **The columns vary per row.** Both SAM2 producers build their record with `if k in res`, so
   two particles in the same call can have different columns — the DataFrame is then the union,
   with NaN where a key was missing for reasons that were never written down.

**D-16 — `bbox` defaults to `()`** while the annotation promises four ints. The `type: ignore` on
that line was written in M2-T02 to expire itself the moment this task fixes it.

**The audit's own yardstick is gone.** Its table measures each producer against the TypeScript
`ParticleMeasurement` interface — and ADR-0012 deleted the frontend. The schema this task
declares has to come from what the science produces and what a consumer needs, not from a
contract that no longer exists.

---

## The decisions this task has to make

### 1. Not one wide table — a core plus declared blocks

| | |
|---|---|
| One superset schema, NaN where a producer cannot fill a column ✅/✗ | Makes `df["height_nm"]` always readable. But it says SEM/TEM *has* heights and they are all missing, and that is a false sentence: the modality cannot produce one |
| **A core every producer emits, plus blocks that are present in full or absent in full** ✅ | `method` says which producer wrote the row, and therefore which blocks to expect. "Absent" stays absent (ADR-0019/0025/0028), and no row inside a table is ever half a block |
| A schema per producer, documented | Is today, with documentation. The reason a consumer cannot write one loop over four tables |

### 2. One name per quantity, and a different name per quantity

- `score` / `sam_score` → **`mask_score`**. Not `score`: this project now has two scores, and
  `Detection.confidence` (ADR-0028) is the detector's. SAM2's is a predicted mask IoU.
- `mask_area_px` → **`area_px`**. The same count of pixels.
- The detector's radius and the mask's radius stop sharing a name:
  **`detector_radius_nm`** (from the blob or box that prompted the measurement — today
  `radius_nm` in the baseline table and `log_radius_nm` in the SAM2 one) and **`radius_nm`**
  (equivalent-circle radius of the mask that was actually measured).

### 3. `bbox` is `None` when there is no box

`tuple[int, int, int, int] | None = None`. `()` is a four-element promise broken silently; a LoG
detection has no box at all, and this milestone has deleted five substitute values already
(ADR-0019, 0024, 0025, 0027, 0028). The `type: ignore` goes with it — `warn_unused_ignores`
makes that automatic.

### 4. Detect mode returns the empty table of the schema it would have filled

Which is the question ADR-0027 left open by name. AFM gets core + height, SEM/TEM gets core +
geometry, both with zero rows.

---

## Scope

**In scope**

1. `core/science/measurement/schema.py` — the column groups, their dtypes,
   `measurement_columns(...)` and `empty_measurement_table(...)`
2. All four producers emit the schema's names; the `if k in res` assembly is replaced by a
   record built from a declared block
3. `Detection.bbox` → `| None`, `type: ignore` removed
4. `run_pipeline`'s detect-mode empty frame
5. Tests, including the two SAM2 producers — which have **no weights here and none in CI**, so
   they are driven by a stub predictor. `_run_sam2_single` is the seam; the same trick M3-T05
   used on `_boxes_to_detections`

**Out of scope**

- **B-059** (`nan <= 0` in `measure_all_baseline`) — still its own commit
- Any change to *how* a height or a geometry is computed. This task moves names and shapes, and
  the golden must show that: no measured value may move
- A typed record class instead of a dict per row. ADR-0027 named it as the single-source
  alternative; it is a refactor of every producer's internals and belongs with the persistence
  layer in M4, where a schema has to be written to SQLite anyway

---

## Expected blast radius, before measuring

- `measure_all_baseline`'s table is golden-recorded on all five AFM phantoms: **column names and
  dtypes move, values must not.** `x_px`/`y_px` become float64 — they hold `int(round(...))`
  today, and a centre is a subpixel quantity that the other three producers already emit as float
- `contracts.default_detection_bbox` — `[]` → `None`, and `default_detection_bbox_len` becomes
  meaningless and goes
- The SAM2 producers are outside the gate entirely (no weights), so their delta is **zero by
  construction and the tests are the whole safety net** — that has to be said, not implied

---

## Definition of done

- [x] One schema module; every producer emits from it, and no producer builds a row with
      `if k in res`
- [x] `mask_score`, `area_px`, `detector_radius_nm` / `radius_nm` — one name per quantity across
      all four
- [x] `Detection.bbox` is `| None`; the `type: ignore` is gone and mypy is happy without it
- [x] Detect mode returns the modality's empty table
- [x] Tests — **31** over five tables, the SAM2 pair driven by a stub predictor
- [x] `make check` green — 392 tests; delta **62 differences, 35 column digests unchanged and 0
      changed**, and the renamed column's digest is byte-identical to its predecessor
- [x] ADR-0031; `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [x] Commit: `M3-T14: one measurement schema, and a bbox that means something`

---

## What it turned up

**The harness had the same bug the code did.** `capture_contracts` called `list(det.bbox)`, which
is a `TypeError` the moment a bbox can be absent — D-16's assumption living inside the tool built
to catch D-16. Fourth time this milestone that the harness was part of the finding.

**The audit missed the worse half of D-17.** It listed each producer's columns against a
TypeScript interface that ADR-0012 has since deleted, which surfaces *missing* and *extra* columns
but not **two quantities sharing a name**. `radius_nm` meaning two different measurements is the
fault that silently corrupts an aggregate, and no column count can see it.

**mypy caught a dead comparison in code written minutes earlier** — `cfg.mode == "segment"` inside
the branch that only runs for `"detect"`. Two errors appeared in this change and both were fixed
rather than annotated.

---

## Notes

The measure of this task is whether a consumer can write one loop over the output of all four
producers. Today it cannot, and the reason is not that the science differs — it is that the same
number has two names and two numbers have one name.
