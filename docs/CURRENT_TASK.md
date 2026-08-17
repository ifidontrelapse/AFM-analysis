# CURRENT TASK

**ID:** `M7-T10`
**Title:** Measurement semantics documented: height, diameter, distance, aspect ratio
**Milestone:** M7 — Annotation & metrology tools, tenth and last task
**Defect:** — · **ADR:** **ADR-0079**
**Branch:** `feat/m7-annotation-tools`
**Status:** **done 2026-08-17.** The record is in `docs/Progress.md` and `docs/TASKS.md`.
**M7's task list is complete; closing the milestone is the operator's call.**

---

## Why this task is next

It is the last task in M7, and the one the milestone earned: M7-T05 and M7-T06 added two outputs *no
algorithm produced*, M7-T09 sent the boxes to a trainer, and M8 will compare a model against numbers
whose definitions live in four producers' docstrings. **A column called `height_nm` is not a
measurement until somebody writes down what it is the height of.**

PROJECT_RULES §8 sets the standard this task has to clear: *"documentation that contradicts the code
is worse than no documentation"* and *"do not claim a feature exists because a document mentions
it — verify in source"*. So the work is reading the four producers and the two hand tools and writing
down **what they actually compute**, including where they disagree.

---

## The decisions this task has to make

**1. One document, `docs/Measurements.md`.**

The semantics are spread across `measurement/schema.py`, `height.py`, `geometry.py`, `metrology.py`
and the two SAM2 producers in `infrastructure`. A docstring can only describe the function it sits
on, and the questions an operator actually asks — *is this height comparable to that one?* — are
about the relationship between them.

**2. The document is checked by a test.**

M5-T03's refrain: *the rule and its enforcement ship together, or only the rule does.* Every column
`measurement_columns` can declare must appear in the document, and the document may name no column
the schema does not have. A column added in M8 then fails a test instead of silently arriving
undocumented.

**3. The project reports radii, not diameters.**

The task's own title says *diameter*, and **there is no diameter column** — there is `radius_px` /
`radius_nm` (the equivalent-area radius of a measured mask) and `detector_radius_nm` (what the
detector that prompted the measurement thought). The document says so and gives the conversion, the
way M7-T07 answered a title that named an operation this project does not perform.

**4. `height_nm` is one column and two estimators, and `method` is the discriminator.**

Both producers compute *peak − baseline*, and neither the peak nor the baseline is the same quantity:

- the baseline producer takes the peak over a **circular mask** built from the detector's sigma, and
  falls back to the **global substrate median** when the ring is too small;
- the SAM2 producer takes it over the **eroded real mask**, and **skips the particle** when the ring
  is too small, so `baseline_source` is always `ring` on that path.

Neither is wrong; a table that mixes them without saying so is.

**5. What was measured is not what was detected.**

Rows are dropped — a mask running off the edge, a non-positive height — so a measurement table is a
subset of the detections (ADR-0033), `particle_id` has gaps, and **B-069** means the column indexes
different things per producer. The document carries that warning where a reader meets the column.

---

## What reading the code turned up

Two defects, both in the geometry block, both **filed rather than fixed** — they change scientific
output, which needs operator sign-off (PROJECT_RULES §4.5), and PROJECT_RULES §4.4 forbids bundling a
numerical fix into another commit:

- **`aspect_ratio` reports `1.0` — the value that means *a circle* — for the most elongated mask
  there is.** A one-pixel-wide line has a minor axis of 0, and the guard substitutes 1.0. Measured:
  a 5×1 line comes back `aspect_ratio=1.0`.
- **`circularity` returns 12.57 for a single pixel**, because a perimeter of 0 is replaced by 1.0.
  Measured. And a *real* digitised disk of radius 10 scores **0.916**, not ~1.0, because skimage's
  perimeter overestimates a rasterised circle — which an operator reading "1.0 = perfect circle"
  will mistake for a finding about their sample.

Both are the defect class this project has removed twice already: **a constant standing in for an
undefined value** (ADR-0025 for scales, ADR-0033 for NaN heights).

---

## Scope

**In scope**

1. `docs/Measurements.md` — every column, both hand tools, units, coordinate conventions, what is
   dropped, and where two producers differ
2. `tests/unit/test_measurement_docs.py` — the document against `measurement_columns`
3. `docs/ProjectFormat.md` — one row pointing at the new document
4. `docs/PROJECT_RULES.md` §0 — the document map gains a row
5. **ADR-0079** — the vocabulary is a contract, and a document nothing checks is a document that drifts
6. **B-071**, **B-072** filed

**Out of scope**

- **Fixing either defect** — operator sign-off, own commit, own ADR, own golden run (§4.4, §4.5)
- **A user manual** — M9-T02. This document is a reference for the numbers, not a guide to the tool
- **New measurements** (volume, tip deconvolution) — B-025, B-026

---

## Definition of done

- [x] `docs/Measurements.md`, verified against the source function by function
- [x] A test that fails when a column is added without documenting it — proved with `volume_nm3`
- [x] ADR-0079 + the ADR index
- [x] B-071 and B-072 filed with the measured values
- [x] `make check` green — 1327 tests, golden byte-identical, mypy unchanged at 6
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_RULES.md`, `ProjectFormat.md`,
      `PROJECT_CONTEXT.md`
- [x] Commit: `M7-T10: what the numbers mean, and where two producers disagree`
