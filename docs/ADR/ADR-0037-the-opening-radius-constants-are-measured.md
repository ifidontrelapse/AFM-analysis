# ADR-0037 — The opening-radius constants are named, exposed and measured

- **Status:** Accepted
- **Date:** 2026-08-08
- **Affects:** `nanoscope/core/science/preprocessing/substrate.py` · **B-064** · M3-T26
- **Numerical impact:** **zero.** The defaults are the current values and the arithmetic is
  unchanged; the golden file is byte-identical, which is the assertion this task's plan made
  before it ran.

## Context

Two numbers set every opening radius in the project:

```python
estimate_rough_radius(..., scale: float = 1.7)                              # a parameter
opening_radius = max(_integer_radius(sizes["typical_radius_px"] * 2.5), 5)  # a bare literal
```

Neither is derived anywhere. The only documentation is the March 2026 docstring — *"a multiplier
so the disk is safely larger than a particle"* — and **both were chosen with the `int()`
truncation ADR-0035 removed still in place**, so whatever tuning they received was done against a
systematically small estimate: the effective margin was `1.7 × int(r)/r`, which at `r = 4.9` is
**1.39**, not 1.7.

Until M3-T15 there was no way to ask whether they were right. Now there is.

## The measurement

Both factors swept over the five AFM phantoms, scored against ground truth with
`evaluate_detections`.

### The rough factor barely matters

| `scale` | mean recall | mean precision | mean localisation | mean radius error |
|---|---:|---:|---:|---:|
| 1.3 | 0.7686 | 0.9958 | 0.5714 px | 0.4986 px |
| 1.5 | 0.7686 | 0.9958 | 0.5714 px | 0.4982 px |
| **1.7** | 0.7686 | 0.9958 | 0.5718 px | **0.4939 px** |
| 2.0 | 0.7686 | 0.9958 | 0.5714 px | 0.5056 px |
| 2.4 | 0.7686 | 0.9958 | 0.5718 px | 0.4943 px |

**Recall and precision are identical across the whole range.** The second stage re-estimates the
radius from Otsu, so the first stage only has to be roughly right — which is M3-T24's finding
approached from the other side, and **it explains why the truncation survived five months**: a
constant whose value does not measurably matter does not get audited.

### The final factor is a genuine trade-off

Per phantom, recall and mean absolute radius error:

| factor | flat | tilted | **dense** | coarse | mean radius error | radii |
|---|---|---|---|---|---:|---|
| 1.5 | 1.000 | 1.000 | **0.886** | 1.000 | 0.890 px | 11, 11, 10, 5, 7 |
| 2.0 | 1.000 | 1.000 | 0.843 | 1.000 | 0.642 px | 15, 15, 13, 6, 9 |
| **2.5** | 1.000 | 1.000 | 0.843 | 1.000 | **0.494 px** | 19, 19, 16, 8, 11 |
| 3.0 | 1.000 | 1.000 | 0.829 | 1.000 | 0.619 px | 22, 22, 19, 9, 13 |
| 4.0 | 1.000 | 0.967 | 0.800 | 1.000 | 0.579 px | 29, 29, 25, 12, 17 |

**A smaller opening finds more particles; a larger one measures their radii better.** The recall
cost lands entirely on `afm_dense_overlapping` — a bigger disk steps *over* two touching particles
instead of into the gap between them — and 1.5 buys three extra detections there for an **80 %
worse** radius error across the set.

`afm_sparse_low_snr` scores **0.000 at every factor**, which is worth stating: its problem is not
the substrate. That is more evidence for **B-062** being a detector-threshold question.

## Decision

**Keep both values. Name them. Expose the literal.**

```python
DEFAULT_ROUGH_SCALE = 1.7
DEFAULT_OPENING_SCALE = 2.5
MIN_OPENING_RADIUS_PX = 5
build_substrate_map(..., opening_scale: float = DEFAULT_OPENING_SCALE)
```

2.5 minimises the radius error on both hard phantoms and is within 0.13 px of the best on the two
easy ones, where the error falls monotonically with the factor. It is the only value in the sweep
that is not beaten on the metric it is best at. 1.7 is kept because nothing in range distinguishes
it.

**The code change is naming and plumbing.** A magic literal inside a branch is not a decision
anyone can revisit — which is exactly why this finding took two tasks to surface. The constant is
now a parameter a caller can sweep, and its docstring carries the table above.

**The `5` floor gets a name too.** It was the third undocumented number on that line, and no
phantom reaches it, so it has never been exercised.

## Consequences

**Positive**

- Three anonymous numbers become three documented ones, each carrying what was measured.
- `opening_scale` is sweepable, so the next person to question it starts from data and a
  parameter rather than from an edit.
- The claim "these constants are reasonable" is now checkable. It was not before.

**Negative**

- The sweep is five values on five synthetic phantoms. **A phantom is not a sample** (ADR-0032),
  so this licenses "2.5 is sound on the phantom set" and nothing about real scans — which is
  **B6** again.
- Keeping a value because a measurement did not beat it is weaker than deriving it. Nothing here
  derives anything; `1.7` and `2.5` remain choices, now informed ones.

**Neutral**

- Zero numerical change. The defaults are the current values.

## What is deliberately not in this commit

**B-067 — a margin that tracks the largest particle rather than the median.** The whole design
picks one radius per image from `typical_radius_px`, a *median*. On a polydisperse sample the
median particle is by definition too small for half of them, and the sweep shows the symptom:
`afm_tilted_polydisperse` is the only phantom that loses a detection at 4.0, and it is the
polydisperse one. A margin derived from the radius distribution's upper tail is a different
algorithm with its own ADR — and, now, a harness that can score it.

**B-062** — unchanged, and reinforced by this sweep.

## The measured delta

**Zero golden differences.** The file is byte-identical and was left untouched rather than
rewritten — `git checkout` on it after the comparison, so the commit carries no phantom change at
all.

That is the whole point of the shape this task took. Naming a constant, exposing it as a
parameter and writing down what it was measured to do are three things that must not move a
number; if any of them had, a default was mistyped. The plan predicted zero and the run returned
zero, which is the second task in a row where the prediction held exactly — and, unlike M3-T24's,
this one was easy to predict, because nothing about the arithmetic changed.

The *measurement* is not in the golden and does not belong there: a five-point sweep across five
phantoms is 25 detection runs, and the golden already costs ten minutes. It lives in this
document, in the constants' own docstrings, and in a test that pins the trade-off's **direction**
rather than its numbers.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Move the final factor to 1.5 | Three more detections on one phantom for 80 % worse radii on four. And the phantom it helps argues for a different substrate *strategy* (B-067), not a smaller disk | Detection count were the only output |
| Move it to 4.0 | Best radius error on the two easy phantoms, worst recall everywhere, and it starts losing particles on the polydisperse one | Every sample were monodisperse and sparse |
| Derive the factor from the radius distribution now | The right answer and a different algorithm. Bundling it here would mean this task both measured the old constant and replaced it, and neither result would be attributable (ADR-0010) | The change were small |
| Leave them as literals, document in the ADR only | A number nobody can pass is a number nobody can re-measure. The plumbing *is* the deliverable | The values were derived rather than chosen |

## Compliance

- `tests/unit/test_opening_scale.py` — **12 tests** in three classes. *The defaults are what they
  were*: the named constants hold 1.7, 2.5 and 5 — asserted literally, so that if they drift the
  sweep in this document stops describing the code — and passing the default explicitly is
  identical to not passing it. *The parameter reaches the opening*: the radius grows monotonically
  with the factor, which is what makes a future sweep a sweep rather than an edit, and the floor
  holds at 0.01. *The trade-off is real*: two particles with a 3 px gap, where a small disk steps
  into the gap and a large one steps over the pair — the mechanism behind
  `afm_dense_overlapping`'s recall falling 0.886 → 0.800, pinned as a property; and the other
  half, that on a sparse field the factor costs nothing.
- Golden: unchanged.

## References

- `ADR-0035` — removed the truncation these constants were chosen against
- `ADR-0032` / M3-T15 — the harness that made the sweep possible, and the reason its result is
  limited to the phantom set
- **B-064**, closed here; **B-067**, filed here
