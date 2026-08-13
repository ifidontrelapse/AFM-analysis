# CURRENT TASK

**ID:** `M6-T06`
**Title:** What the run says about the sample, not about one particle
**Milestone:** M6 — Analysis workflow in the GUI, sixth task
**Defect:** — · **ADR:** **ADR-0066**
**Branch:** `feat/m6-analysis-workflow`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M6-T05 put thirty rows on screen. Nobody measures thirty particles to read thirty numbers — they
measure thirty to say *"the particles on this sample are 15 nm across, give or take 4"*. That
sentence is the reason the table exists, and nothing in the application can produce it.

---

## The decisions this task has to make

**1. Who computes the statistics?** `application`, over the stored table — not the widget.

A panel that computes a mean is a second place where "what is the mean of this column" is decided,
and the moment a column contains a `NaN` the two answers differ. It is also the rule this milestone
runs on: **the UI introduces nothing of its own.**

**2. How many bins?** By a **named rule**, not by a number somebody liked.

`numpy`'s `"auto"` — the larger of Sturges and Freedman–Diaconis — because a fixed 20 is an invented
parameter, and the shape of a histogram *is* the claim it makes. An operator who wants a different
binning is asking a different question, and that is a control, not a default.

**3. What is drawn, and by what?** Bars, painted by Qt.

matplotlib lives in `infrastructure` and `gui/` may not import it (Architecture §3.2, checked by a
test since M5-T06). The binning is `numpy` in `application`; what is left for the widget is
rectangles, which is thirty lines of `paintEvent` and no new dependency. `QtCharts` would be a whole
module to draw a bar chart nobody will interact with.

**4. Which columns?** The numeric ones the table actually has, physical first.

`particle_id` is an identifier, not a measurement, and averaging it is a number with no meaning.
Which columns exist depends on the producer (ADR-0031) and on whether the scale was known
(ADR-0025) — so the list is derived from the table in hand, never assumed.

**5. What does an absent column look like?** Absent.

A column that is entirely `NaN` — every `_nm` column of an unscaled scan — is not offered, and the
panel says the scale is unknown rather than showing `nan ± nan`.

---

## Scope

**In scope**

1. `application/use_cases/statistics.py` — `numeric_columns`, `summarise`, `histogram`
2. `gui/panels/statistics.py` — the column choice, the numbers, and the bars
3. `MainWindow` — the Statistics dock, beside the measurements
4. **ADR-0066** — statistics in `application`, a named binning rule, Qt-painted bars
5. Tests: the numbers against `numpy` on the same column, `NaN` excluded, the bin rule, an unscaled
   run offering no physical column, and the panel following the run

**Out of scope**

- **Statistics across images** — M6-T08's navigation has to exist first, and "the sample" is then a
  different question from "this scan"
- **Fitting a distribution** — a log-normal fit is a scientific claim with an ADR behind it
- **Exporting the plot** — M6-T07 exports the table; a figure is `infrastructure.imaging.plots`

---

## Definition of done

- [x] The summary matches `numpy` on the same column, with `NaN` excluded
- [x] The histogram's bins come from a named rule
- [x] An unscaled run offers no physical column and says why
- [x] ADR-0066 + the ADR index
- [x] `make check` green — 1127 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M6-T06: what the run says about the sample, not about one particle`

---

## What it turned up

**An unscaled scan keeps its heights.** The first wording of the panel's note said an unknown scale
means "no physical columns", and the test written to prove it failed: `height_nm`, `peak_nm`,
`baseline_nm` and `mean_nm` are all there. **A height is calibrated by the z axis; a radius comes
from the pixel size.** Losing the lateral scale costs the sizes and nothing else — the note says
that now, and the test asserts both halves.

**`np.issubdtype` raises on a pandas extension dtype.** Every measurement table carries a string
column (`method`, ADR-0031), and asking numpy whether it is numeric is a `TypeError` rather than a
"no". The check is `dtype.kind` now.
