# CURRENT TASK

**ID:** `M3-T11`
**Title:** Unknown pixel scale (`None`) must not crash either detector
**Milestone:** M3 — Numerical correctness, sixth task
**Defect:** **D-07** (high) · **ADR:** **ADR-0019**
**Branch:** `sci/unknown-scale` (stacked on `sci/log-zero-max`)
**Status:** **done 2026-08-05.** Rewritten for the next task at the start of the next
session; the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task was next

The highest-severity unblocked defect left, and the one with a reachable user-facing route: any
SEM or TEM image without scale metadata went straight to a `TypeError`. It is also the
precondition for **M3-T20** — that task stops the npy loader inventing a scale of `1.0`, which
would push `None` into a path that, until this commit, crashed on it.

---

## Scope

**In scope**

1. `detect_particles`, `LogDetector.detect`, `YoloDetector.detect` and both conversion helpers:
   accept `pixel_size_nm: float | None`
2. `Detection.radius_nm: float | None`, and the `NaN` → `None` mapping at the entity boundary
3. `Detector` port and `BaseDetector` signatures, so the contract says it too
4. `run_pipeline`'s `nm_per_pixel` annotated `float | None` — the mypy error that *was* this
   defect
5. The harness: `detect_particles_no_scale` and `boxes_to_detections_{scaled,no_scale}`
6. **ADR-0019**, including why this `NaN` is not the `NaN` ADR-0018 removed

**Out of scope** — all four listed with reasons in ADR-0019 §"What is deliberately not in this
commit"

- **D-20 / M3-T20** — `load_afm(fmt="npy")` fabricating `pixel_size_nm or 1.0`
- `build_substrate_map` and the preprocessing chain, which divide by the scale. Unreachable with
  `None` until M3-T20 lands
- `plot_detections`, called only from the notebooks, always with an AFM scan
- `run_sam2_from_blobs`'s `if nm_per_pixel else None`, which also swallows an explicit `0.0`

---

## Definition of done

- [x] Both detectors accept `pixel_size_nm=None` and return detections
- [x] `radius_nm is None`, never `0.0`, never `radius_px`
- [x] Pixel-space output bit-identical with and without a scale
- [x] The SEM route through `run_pipeline` completes
- [x] `make check` green — 159 tests
- [x] Delta quantified: **168 golden keys added, 0 changed**; mypy **19 → 18**
- [x] ADR-0019; `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M3-T11: an unknown pixel scale is a state, not a crash`

---

## The delta

| what | before | after |
|---|---|---|
| `detect_particles_no_scale` · 5 AFM phantoms | `TypeError` | 24 blobs, `radius_nm` all NaN, **0** detections carrying a radius |
| `boxes_to_detections_{scaled,no_scale}` · 7 phantoms | not recorded | `[5.0, 9.5]` vs `[null, null]`, identical `radius_px` |
| mypy errors | 19 | **18** |

No recorded number moves: every phantom has a scale, so the working path is byte-identical.

---

## What it turned up

**mypy had been reporting this defect since M1-T04 and nobody read it that way.**
`pipeline.py:62 — Incompatible types in assignment (expression has type "float | None", variable
has type "float")` is D-07, stated at the assignment instead of at the crash. It sat in the
19-error baseline among genuinely legacy noise. The lesson is not "fix the baseline" — it is that
a non-zero tolerated baseline hides the entries that are defects, and M2-T12's job of driving it
to zero is worth more than it looks.

---

## Notes for the next session

**`M3-T20` is next**, and it is the other half of this one: the npy loader's
`pixel_size_nm or 1.0` and `scan_size_nm or float(z.shape[0])` fabricate a physical scale from a
row count. `None` is now survivable in the detectors, but **not yet in `build_substrate_map`**,
which divides by the scale — so M3-T20 must carry that guard or state why it does not.

**M3-T12** (D-08, empty measurements return a zero-column DataFrame) and **M3-T17** (the SPM
parser's no-`Scan Size` fallback) are the other unblocked `high` ones. T17 and T20 are the same
file.

**Four decisions still block five tasks: B2, B3, B4, B7.**
