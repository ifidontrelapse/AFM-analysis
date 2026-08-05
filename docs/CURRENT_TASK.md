# CURRENT TASK

**ID:** `M3-T06`
**Title:** Otsu sizing — raise on empty-after-filter, report post-filter `n_objects`
**Milestone:** M3 — Numerical correctness, fourth task
**Defects:** **D-05** (high) · **D-06** (medium) · **ADR:** **ADR-0017**
**Branch:** `sci/otsu-sizing` (stacked on `sci/yolo-letterbox`)
**Status:** **done 2026-08-05.** Rewritten for the next task at the start of the next
session; the record is in `docs/Progress.md` and `docs/TASKS.md`.

> **Two defects, one commit.** They are the same eight lines of `estimate_radius_otsu` — the
> filter and what is reported about it — and separating them would mean touching those lines
> twice for one intent. ADR-0010 forbids bundling a numerical fix with a *refactor*, not
> fixing two halves of one broken step.

---

## Why this task was next

D-05 is the highest-severity unblocked defect left, and it is the one that produces the
project's worst kind of failure: a `nan` created in one function and reported, as a different
error, in another. Everything else in M3 is either blocked on an operator decision (D-04,
D-12, and now the tiling question) or lower severity.

---

## Scope

**In scope**

1. `estimate_radius_otsu`: raise when the size filter empties the set
2. The same function: `n_objects` counts survivors
3. The harness records D-05's own reproduction, so the golden holds the error
4. **ADR-0017**, including why the message carries the largest object measured

**Out of scope**

- **D-04 / B2** — `min_size_pixel` flooring to zero, the reason the new error is mostly
  unreachable today. Open operator decision
- **M3-T13** — the typed error taxonomy that will re-home this `ValueError`
- The duplicated `radii_nm` assignment two lines below. Left alone on purpose: this file's
  commits are numerical, and tidying is not this commit's intent

---

## Definition of done

- [x] Empty-after-filter raises, naming the parameter, its value and the largest object
- [x] `n_objects == len(radii_px)`, always
- [x] The golden records the error instead of recording nothing
- [x] `make check` green — 140 tests
- [x] Delta quantified: **8 golden differences**
- [x] ADR-0017; `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M3-T06: fail loudly when the size filter empties the set`

---

## The delta

| what | before | after |
|---|---|---|
| `afm_sparse_low_snr` · `n_objects_reported` | **1023** | **75** |
| `degenerate_inputs.extreme_aspect` · error | `cannot convert float NaN to integer`, in `build_substrate_map` | the sizing's own message, in `estimate_radius_otsu` |
| `estimate_radius_otsu_all_filtered` | — | added, 5 phantoms |

Four of five AFM phantoms did not move, and that is D-04's doing: `min_size_pixel` floors to
0 on coarse scans, so the filter removes nothing and both counts already agreed. This fix
starts mattering on real data the day **B2** is answered.

---

## What it turned up

**An M3-T01 test had been passing because of the defect.**
`test_a_different_radius_produces_a_different_substrate` uses four 4.7 px particles and
passed `min_size_nm=5` at 1 nm/px — the filter removed all four, the sizing returned `nan`,
and the test never looked at `sizes`. It now passes `min_size_nm=1` and says why. A `nan` in
a field nobody reads is exactly how D-05 stayed invisible.

---

## Notes for the next session

**`M3-T07`** (D-11, LoG normalisation against a zero maximum) is the next unblocked task.

**`M3-T21` is now blocked on B7.** The single-crop tiling cannot be fixed by engineering
alone: making it tile means either upsampling the scan to ≥ 1120 px (the model then sees
interpolated pixels), using crops smaller than 640 (the model upscales each instead), or
accepting that tiling is pointless at 512 px and dropping the backend. That is a cost-versus-
resolution trade-off about real samples.

**Four decisions now block five tasks: B2, B3, B4, B7.** M3 cannot close without them.
