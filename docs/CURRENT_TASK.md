# CURRENT TASK

**ID:** `M3-T02`
**Title:** The minimum particle size is a physical size
**Milestone:** M3 — Numerical correctness, eighth task
**Defect:** **D-04** (critical) · **Decision:** **B2** · **ADR:** **ADR-0024**
**Branch:** `sci/min-size-in-nm` (stacked on `sci/detection-polarity`)
**Status:** **done 2026-08-06.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

D-04 is one of the two `critical` defects the audit left open, and the last one that needed an
operator answer. B2 was answered on 2026-08-05: **filter in nanometres, delete the `int()`**.
It goes after B3/M3-T10 because M3-T10 changes which pixels are particles and this one changes
which particles are kept — measuring them in that order keeps the two deltas attributable.

The defect, in one line:

```python
min_size_pixel = int(min_size_nm / pixel_size_nm)     # int(5 / 9.77) == 0
```

`pixel_size_nm` across the operator's 120 real scans: min 1.95, **median 9.77**, max 29.30. With
the default `min_size_nm = 5`, **108 of 120 scans (90 %) get a threshold of 0**, so the noise
filter in `estimate_radius_otsu` admits every connected component, single-pixel noise included.
Those radii set `typical_radius_px`, which sets the opening radius *and* the LoG sigma range.

---

## Scope

**In scope**

1. `estimate_radius_otsu` takes `min_size_nm` and compares `radii_nm >= min_size_nm`
2. `estimate_rough_radius` takes `min_size_nm` and converts it to pixels **without** `int()`
3. `build_substrate_map` passes its own `min_size_nm` through, unconverted, at all three sites
4. The error message speaks nanometres, because the caller set a physical minimum (ADR-0017's
   message, same shape, new units)
5. The harness records `min_size_nm_used`, `min_size_px_equivalent` and `min_size_px_floored` —
   the last one is the old arithmetic, kept as the measuring stick for what was lost
6. **ADR-0024**

**Out of scope**

- `min_size_nm` as a `PipelineConfig` field. It is a `build_substrate_map` default (5) today and
  promoting it is a config change, not a numerical one — M3-T13's neighbourhood
- The GUI's spin box for it. There is no GUI yet
- Any change to *how* radii are measured. `equivalent_diameter_area / 2` is untouched

---

## Definition of done

- [x] No `int()` between `min_size_nm` and any comparison
- [x] The threshold means the same physical thing at every pixel scale
- [x] Unit tests, one per pixel-scale regime; restoring the `int()` turns 3 of 5 red
- [x] `make check` green — 204 tests
- [x] Delta quantified: **47 differences — 27 changed, 15 added, 5 removed**; mypy **15 → 15**
- [x] ADR-0024; `STATE.md`, `Progress.md`, `TASKS.md`, `Roadmap.md`, `Architecture.md`,
      `PROJECT_CONTEXT.md`, ADR index
- [x] Commit: `M3-T02: the minimum particle size is a physical size (B2)`

---

## The delta

| phantom | nm/px | old | new | objects kept |
|---|---|---|---|---|
| `afm_flat_monodisperse` | 2.00 | 2 px | 2.5 px | 24 → 24 |
| `afm_coarse_pixels` | 9.77 | **0 px** | 0.512 px | 14 → 14 |
| `afm_dense_overlapping` | 2.00 | 2 px | 2.5 px | 51 → 51 |
| `afm_tilted_polydisperse` | 2.00 | 2 px | 2.5 px | 29 → 29 |
| `afm_sparse_low_snr` | 2.00 | 2 px | 2.5 px | **75 → 17** |

No height moves anywhere: the final opening radius is 8 before and after on the one phantom whose
radii change, so `substrate` and `z_above` are byte-identical.

---

## What it turned up

**The phantom built for D-04 does not move.** `afm_coarse_pixels` is the 9.77 nm/px case whose
`min_size_pixel_used: 0` the characterization baseline records as the defect's fingerprint — and
its numbers are unchanged, because the smallest object a labelling can produce is one pixel,
whose equivalent radius is 5.51 nm there. Re-reading all 628 scan headers put a number on it:
the zero threshold was **inconsequential on 58 % of the scans**, disabled a working filter on
**32 %**, and the finest **10 %** were hurt by *truncation* rather than by the floor — which is
`afm_sparse_low_snr`'s 2.5 px → 2 px, and 58 of its 75 objects.

**mypy did not move**, for the first time in three numerical tasks. `int(float) -> int` is
perfectly typed; a unit error has no static shadow, and the `_nm` / `_px` convention is the only
check this class of defect has.

**Three tests written earlier in this task were replaced.** Each asserted something the *old*
code also satisfied — a test that cannot fail on the defect it names.

---

## The duplicated `radii_nm`

`radii_nm = radii_px * pixel_size_nm` was written twice in a row, identically — the audit's
§Duplication entry, and deliberately untouched by M3-T01, M3-T06 and M3-T09 because their commits
were numerical and ADR-0010 keeps tidying out of them. It is fixed **here** rather than in a
tidying commit because this task *has* to move that line: the filter now needs `radii_nm` before
it runs, so the assignment moves above the filter and the second copy has nowhere left to be.
One line, forced by the change, not swept in beside it.

---

## Notes for the next session

Remaining from the operator's answers, in order:

1. **B6 → M3-T16** — header-only SPM fixtures
2. **B-040** — purge `node_modules` and the weights from git history. **Last**, because it
   rewrites every SHA above

Then the unblocked `high` ones: **M3-T20**, **M3-T12**, **M3-T17** (T17 and T20 are the same
file and are worth reading together).
