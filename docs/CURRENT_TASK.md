# CURRENT TASK

**ID:** `M3-T07`
**Title:** Guard the LoG normalisation against a non-positive maximum
**Milestone:** M3 — Numerical correctness, fifth task
**Defect:** **D-11** (medium) · **ADR:** **ADR-0018**
**Branch:** `sci/log-zero-max` (stacked on `sci/otsu-sizing`)
**Status:** **done 2026-08-05.** Rewritten for the next task at the start of the next
session; the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task was next

The three `high` defects above it are either blocked on an operator decision (D-12 on B3,
M3-T21 on B7) or untouched by this session's reading. D-11 sits in the file M3-T06 had just
walked into: the `nan` it produced was the same *kind* of failure — a number created in one
function, reported as something else in another — and the fix was already written, once, in a
sibling function of the same module.

---

## Scope

**In scope**

1. `estimate_log_threshold_adaptive`: return the default threshold on a non-positive maximum
2. `detect_particles`: return an empty `(0, 4)` on a non-positive maximum, after `sizes` is
   validated
3. `DEFAULT_THRESHOLD = 0.05` named once instead of written three times
4. The harness: a `negative_with_structure` degenerate input, and scalars recorded as numbers
   instead of the string `"non-array"` — without both, the fix is invisible in the golden
5. **ADR-0018**, including why this returns where ADR-0017 raises

**Out of scope**

- **D-12 / B3** — detection polarity, the reason a negative map reaches the detector at all.
  Open operator decision
- **M3-T19** — `responses` rebinding from `list[float]` to ndarray in the same function. A
  typing defect, not a numerical one; ADR-0010 forbids the bundle
- Changing the normalisation itself (`ptp`, `abs`, clipping). Each moves every recorded
  detection and needs its own ADR — the table in ADR-0018 says why none was taken

---

## Definition of done

- [x] Both division sites stop on a zero, negative or `nan` maximum
- [x] `detect_particles` returns zero particles rather than raising, and says why in the log
- [x] The adaptive threshold is always in `(0, 1]`
- [x] The harness records the number that was wrong
- [x] `make check` green — 151 tests
- [x] Delta quantified: **65 golden keys added, 0 changed**
- [x] ADR-0018; `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M3-T07: require a positive maximum before normalising`

---

## The delta

| what | before | after |
|---|---|---|
| `negative_with_structure` · `estimate_log_threshold_adaptive` | **2.4997**, unrecorded | **0.05** |
| `estimate_log_threshold_adaptive` | recorded nowhere | recorded on all 11 degenerate inputs |
| `negative_with_structure` | — | added |

**Zero numbers changed.** `build_substrate_map` guarantees `z_above >= 0`, so every phantom and
every scan through the normal path has a positive maximum and comes out byte-identical. The
negative case reaches the detector only through `LogDetector.detect` on a raw SEM/TEM image,
which is D-12.

---

## What it turned up

**The harness had been recording the wrong thing since Phase 0.** `capture_degenerate` wrote
every non-array result down as the literal string `"non-array"`, so a threshold of 2.4997 —
outside the interval it is compared against — was captured, discarded, and stored as prose. And
the only negative degenerate input was a *constant* −5, which survives the division looking
like a constant. Ten inputs recorded D-11 and none of them could show it.

That is the third task in M3 whose harness change is larger than its code change (M3-T01,
M3-T04, now this one). The pattern is worth naming: **a golden that cannot fail on a defect is
not evidence the defect is absent.**

---

## Notes for the next session

The unblocked `high` defects left in M3 are **M3-T11** (D-07, unknown pixel scale crashes both
detectors), **M3-T12** (D-08, empty measurements return an unstable schema), **M3-T17** (the
SPM parser's no-`Scan Size` fallback crashes) and **M3-T20** (`load_afm(fmt="npy")` fabricates
a physical scale). T17 and T20 are the same file and arguably the same defect — read them
together before splitting them into two commits.

**Four decisions still block five tasks: B2, B3, B4, B7.** M3 cannot close without them.
