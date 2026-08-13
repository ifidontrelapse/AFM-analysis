# CURRENT TASK

**ID:** `M7-T06`
**Title:** The heights under a line, and what the notebook actually did
**Milestone:** M7 — Annotation & metrology tools, sixth task
**Defect:** — · **ADR:** **ADR-0075**
**Branch:** `feat/m7-annotation-tools`
**Status:** **done 2026-08-14.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M7-T05 stored a line and read it as a length. A profile is the same geometry read as **the heights
underneath it** — the row already carries `kind`, and the migration was written for this.

M7's third exit criterion names its own reference:

> *"A height profile along a drawn line matches the notebook implementation."*

---

## The decisions this task has to make

**1. What the notebook actually did, and what that means for the criterion.**

`notebooks/afm_gold_nanoparticles.ipynb` §5.1 takes `z_flat[y_i, x_i-half : x_i+half]` — **a
horizontal row slice**, no interpolation, plotted against `(arange(2*half) - half) * pixel_size_nm`.

So the criterion covers exactly one case: an axis-aligned line. That is the case the compliance test
can assert *equality* for, and it is asserted. **An arbitrary line has no notebook counterpart**, so
the general case is an extension with a rule of its own — which is the honest version of "matches
the notebook".

**2. How an arbitrary line is sampled.** Bilinear, one sample per pixel of length.

A diagonal profile made of nearest-neighbour steps is a picture of the sampling, not of the sample.
One sample per pixel of length means a horizontal line returns **exactly the pixels it crosses**,
which is what makes §1's equality hold rather than approximately hold.

**3. Which array is profiled?** The stage on screen — and the panel says which.

Profiling a raw map and a flattened one give different numbers, and both are legitimate questions.
ADR-0061 already made the stage visible; a profile that did not name its stage would be a
measurement whose provenance is a checkbox somebody set four clicks ago.

**4. It is a numerical entry point, so it validates like one.**

`ensure_height_map` at the door (ADR-0030, its fifteenth site). A profile of a 3-D array or of a NaN
map is a wrong answer waiting to be plotted.

**5. Distances in nm only with a scale** — ADR-0025, the same as M7-T05's length.

---

## Scope

**In scope**

1. `core/science/metrology.py` — `height_profile`, validated and tested against the notebook's slice
2. `application/use_cases/metrology.py` — the profile of a ruler over a stage array
3. `gui/panels/profile.py` — the plot, painted by Qt like M6-T06's histogram
4. `gui/` — the profile tool (the `kind` M7-T05 already stores) and the dock
5. **ADR-0075** — what the notebook did, bilinear sampling, and the named stage
6. Tests: equality with the notebook's row slice, the bilinear case, a validated input, an unknown
   scale, and the plot following the selection

**Out of scope**

- **Exporting a profile** — a CSV of one line is M9's packaging question, and nobody has asked
- **Roughness or step-height statistics over a profile** — each is a scientific claim with its own
  ADR

---

## Definition of done

- [x] A horizontal profile equals the notebook's row slice, asserted as equality
- [x] A diagonal one is bilinear, one sample per pixel of length
- [x] The plot names the stage it measured
- [x] ADR-0075 + the ADR index
- [x] `make check` green — 1250 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M7-T06: the heights under a line, and what the notebook actually did`

---

## What it turned up

**The exit criterion names a reference that covers one case.** The notebook's "height profile" is a
horizontal row slice through a particle — no interpolation, no arbitrary line. So *"matches the
notebook implementation"* can be asserted as an **equality** for the axis-aligned case and cannot be
asserted at all for the case the tool actually offers. Both halves are written down: the equality is
a test, and the extension is a decision with a rule and a test of its own. Reading the reference
before implementing against it turned a vague criterion into two precise ones.
