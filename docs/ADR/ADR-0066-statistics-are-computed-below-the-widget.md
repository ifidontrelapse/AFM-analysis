# ADR-0066 — Statistics are computed below the widget

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M6-T06)
- **Affects:** `application/use_cases/statistics`, `gui/panels` · M6

## Context

M6-T05 put thirty rows on screen. Nobody measures thirty particles to read thirty numbers — they
measure thirty to say *"the particles on this sample are 15 nm across, give or take 4"*, and nothing
in the application could produce that sentence.

## Decision

### 1. `application` computes; the widget renders

A panel that computes a mean is a second place where *"what is the mean of this column"* is decided,
and the first `NaN` makes the two answers differ. It is also this milestone's rule: **the UI
introduces nothing of its own.**

`count` is the number of **finite** values, not the number of rows. The spread is the **sample**
standard deviation (`ddof=1`) — these are particles measured *from* a sample — and it is undefined
for one particle, which the panel prints as absent rather than as a zero somebody could mistake for
a measurement.

### 2. The bins come from a named rule

`numpy`'s `"auto"`, the larger of Sturges and Freedman–Diaconis. A fixed 20 would be an invented
parameter, and **the shape of a histogram is the claim it makes**. An operator who wants different
binning is asking a different question, and that is a control rather than a default.

### 3. The bars are painted by Qt

matplotlib lives in `infrastructure` and `gui/` may not import it (Architecture §3.2, enforced since
M5-T06). The binning is `numpy`'s, in `application`; what is left for the widget is rectangles —
thirty lines of `paintEvent`, no new dependency, and no `QtCharts` module to draw a chart nobody
interacts with.

### 4. The columns are the table's own, identifiers excluded

Derived from the table in hand, because which columns exist depends on the producer (ADR-0031) and
on what was known about the scan. `particle_id` is not offered: averaging an identifier produces a
number with no meaning. A column with nothing finite in it is not offered either — `nan ± nan` is
not a statistic.

### 5. An unknown scale costs the **sizes**, not the heights

The panel says *"the lateral scale is unknown, so sizes in nanometres are absent; heights are
calibrated by the z axis and are not affected."*

This is the correction the task turned up: a height comes from the z calibration and stays in
nanometres without any lateral scale, while a radius comes from the pixel size and is absent
(ADR-0025). "No physical columns" — the first wording — would have been wrong about half the table.

## Consequences

**Positive** — the run can be described as a distribution without leaving the application; the
numbers are testable against `numpy` on the same values, which is the only assertion that means
anything here; a widget cannot disagree with an export about what the mean is.

**Negative** — the statistics are per **run**, not per sample: describing a dataset needs M6-T08's
navigation first, and "the sample" is then a different question from "this scan". The default column
is the first physical one alphabetically, which is a rule rather than a judgement — `baseline_nm`
before `height_nm` is not what an operator would pick first, and the trigger for changing it is one
of them saying so.

**Neutral** — no distribution fit. A log-normal over particle radii is a scientific claim, and it
gets an ADR and a test against the evaluation harness before it gets a curve on a chart.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Compute in the panel | A second place deciding what a mean is, differing at the first `NaN` |
| A fixed bin count | An invented parameter, and the histogram's shape is its claim |
| `QtCharts` | A whole module for bars nobody clicks |
| matplotlib in the widget | `gui/` may not import `infrastructure`, and a test says so |
| Offer every numeric column | An averaged `particle_id` is a number with no meaning |
| Say "no physical columns" without a scale | Wrong about heights, which the z axis calibrates |

## Compliance

`tests/gui/test_statistics.py` checks the summary against `numpy` over the finite values, that one
particle has no spread, that identifiers and all-`NaN` columns are not offered, that the bins are
`np.histogram(..., bins=BIN_RULE)`'s own, and that an unscaled run **keeps its heights and loses its
sizes**.

## References

- ADR-0031 — the schema whose columns this reads
- ADR-0025 — the absent lateral scale of §5
- ADR-0033 — why `count` is finite values rather than rows
- `docs/Roadmap.md` M6 — *the UI must not introduce its own defaults*
