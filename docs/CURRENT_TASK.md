# CURRENT TASK

**ID:** `M3-T09`
**Title:** Opening radii are integers, rounded up
**Milestone:** M3 — Numerical correctness, seventh task
**Defect:** **D-10** (medium) · **Decision:** **B4** · **ADR:** **ADR-0020**
**Branch:** `sci/opening-radius-ceil` (stacked on `sci/unknown-scale`)
**Status:** **done 2026-08-05.** Rewritten for the next task at the start of the next
session; the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task was next

The operator answered B2, B3, B4, B6 and B7 in one sitting, unblocking five tasks. B4 is the
smallest of them and touches the file the next two also touch, so it goes first: its delta is
attributable on its own, and M3-T02 (B2) then measures against a substrate that already rounds
consistently.

---

## Scope

**In scope**

1. `_integer_radius` — one `ceil`, in `get_substrate_map`, which every caller passes through
2. `estimate_rough_radius` returns the `int` it has always been annotated to return, on both exits
3. `build_substrate_map` reports the integer it used, on both branches
4. **ADR-0020**, including why "round to the nearest odd" is unnecessary

**Out of scope**

- **M3-T02 / B2** — `min_size_pixel` flooring to zero. Next task, and it moves the same numbers
  again; ADR-0010 keeps the two deltas separate
- The duplicated `radii_nm` assignment. Still deliberately untouched: this file's commits are
  numerical

---

## Definition of done

- [x] Every radius reaching `disk()` is an integer
- [x] Rounding is up, at one site, and the reported radius equals the used radius
- [x] `make check` green — 170 tests
- [x] Delta quantified: **696 golden values, 0 keys added**; mypy **18 → 15**
- [x] ADR-0020; `STATE.md`, `Progress.md`, `TASKS.md`, ADR index
- [x] Commit: `M3-T09: opening radii are integers, rounded up (B4)`

---

## The delta

| phantom | opening radius | blobs (true) | mean height nm |
|---|---|---|---|
| `afm_flat_monodisperse` | 17 → **19** | 24 → 24 (24) | 16.1202 → 16.1194 |
| `afm_coarse_pixels` | 9 → **11** | 14 → 14 (14) | 17.8636 → 17.8664 |
| `afm_dense_overlapping` | 14 → **16** | 59 → 59 (70) | 13.3297 → **13.3791** |
| `afm_tilted_polydisperse` | 17 → **18** | 30 → 30 (30) | 16.1175 → 16.1030 |
| `afm_sparse_low_snr` | 7 → **8** | 0 → 0 (6) | — |

No particle count moves. The largest height change is **0.049 nm (0.37 %)**.

---

## What it turned up

**The 696 changed values are propagation, not magnitude.** A reader who sees the biggest golden
delta in M3 and concludes D-10 was the biggest defect in M3 would be wrong: the radius feeds the
substrate, the substrate feeds `z_above`, and every measurement is taken against it. The defect
itself is worth 0.05 nm. It was worth fixing because it was silent and systematic.

**Three of mypy's errors were this defect, stated statically, since M1-T04.** Second task in a
row where that is true (M3-T11 found `pipeline.py:62` the same way). A tolerated non-zero mypy
baseline is not neutral — it hides the entries that are defects.

---

## Notes for the next session

Four more operator answers are waiting to be executed, in this order:

1. **B7 → M3-T21** — `use_tiling=False` becomes the default; the tiled backend has never tiled
2. **B-058** — an ADR for the golden storing CPython exception text, before any Python upgrade
3. **B3 → M3-T10** — explicit polarity per modality; `Polarity` already exists in `core/values`,
   adopted by nothing since M2-T02
4. **B2 → M3-T02** — the critical one: filter in nanometres, `int()` deleted
5. **B6 → M3-T16** — header-only SPM fixtures
6. **B-040** — purge `node_modules` and the weights from git history. **Last**, because it
   rewrites every SHA above

**B-054** (two README figures over 1 MB) is closed by operator decision: the README is rewritten
in M9-T01 and the figures go with it.
