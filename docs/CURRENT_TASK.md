# CURRENT TASK

**ID:** `M3-T26`
**Title:** The opening-radius constants are named, exposed, and measured
**Milestone:** M3 — Numerical correctness, twenty-fourth task
**Defect:** **B-064** (filed by M3-T24) · **ADR:** **ADR-0037**
**Branch:** `sci/m3-numerical-correctness` (the consolidated branch — see the declared
deviation from PROJECT_RULES §7 in `STATE.md`)
**Status:** **done 2026-08-08.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

The last engineering finding open in M3. `M3-T16` waits on **B6**, **B-062** wants an operator's
view, **B-065** and **B-066** were filed an hour ago and each needs a decision this does not.

Two numbers set every opening radius in the project:

```python
estimate_rough_radius(..., scale: float = 1.7)          # a parameter, undocumented beyond one line
opening_radius = max(_integer_radius(sizes["typical_radius_px"] * 2.5), 5)   # a bare literal
```

Neither is derived anywhere. The only documentation is the March 2026 docstring — *"a multiplier
so the disk is safely larger than a particle"* — and **both were chosen with the `int()`
truncation M3-T24 removed still in place**, so whatever tuning they got was done against an
estimate that was systematically small: the effective margin was `1.7 × int(r)/r`, which at
r = 4.9 is **1.39**, not 1.7.

What makes this a task rather than a preference is that **M3-T15 can now score a candidate.**

---

## What the measurement says

Swept over the five AFM phantoms, scoring against ground truth.

### The rough factor barely matters

| `scale` | mean recall | mean precision | mean localisation | mean radius error |
|---|---:|---:|---:|---:|
| 1.3 | 0.7686 | 0.9958 | 0.5714 px | 0.4986 px |
| **1.7** | 0.7686 | 0.9958 | 0.5718 px | **0.4939 px** |
| 2.4 | 0.7686 | 0.9958 | 0.5718 px | 0.4943 px |

**Recall and precision are identical from 1.3 to 2.4.** The response is flat because the second
stage re-estimates from Otsu, which is what M3-T24 measured from the other direction — and it is
*why nobody noticed the truncation for five months*. A constant whose value does not matter does
not get audited.

### The final factor is a real trade-off

| factor | dense recall | flat / tilted / coarse recall | mean radius error | radii |
|---|---:|---|---:|---|
| 1.5 | **0.886** | 1.000 | 0.890 px | 11, 11, 10, 5, 7 |
| 2.0 | 0.843 | 1.000 | 0.642 px | 15, 15, 13, 6, 9 |
| **2.5** | 0.843 | 1.000 | **0.494 px** | 19, 19, 16, 8, 11 |
| 3.0 | 0.829 | 1.000 | 0.619 px | 22, 22, 19, 9, 13 |
| 4.0 | 0.800 | 0.967 | 0.579 px | 29, 29, 25, 12, 17 |

**A smaller opening finds more particles; a larger one measures their radii better.** The recall
cost is entirely on `afm_dense_overlapping` — a bigger disk steps over two touching particles as
one — and 1.5 buys three more detections there for an **80 % worse** radius error.

**2.5 minimises the radius error on both hard phantoms** (`tilted` 0.718, `dense` 0.642) and is
within 0.13 px of the best on the two easy ones, where the error falls monotonically with the
factor. It is also the only value in the sweep that is not beaten on the metric it is best at.

---

## The decision this task has to make

**Keep both values — and stop them being anonymous.**

| | |
|---|---|
| **Keep 1.7 and 2.5, name them, expose the second, record the measurement** ✅ | The sweep says 2.5 sits at the optimum of the metric that varies and the rough factor's value is nearly irrelevant. A constant that survives measurement is worth more than one nobody questioned |
| Move the final factor to 1.5 | Buys 3 detections on one phantom for 80 % worse radii on four. And the phantom it helps is the one that argues for a *different* substrate strategy, not a smaller disk |
| Make them adaptive — track the largest particle, not the median | The right question for a polydisperse sample, and it is a different algorithm with its own ADR. **Filed as B-067**, with the measurement that motivates it |

**What changes in the code:** the bare `2.5` becomes a named, documented parameter, matching the
rough factor that already is one. A magic literal inside a branch is not a decision anyone can
revisit — the whole reason this finding took two tasks to reach.

---

## Scope

**In scope**

1. `DEFAULT_ROUGH_SCALE = 1.7` and `DEFAULT_OPENING_SCALE = 2.5` as named module constants, each
   documented with what the sweep measured
2. `build_substrate_map(..., opening_scale: float = DEFAULT_OPENING_SCALE)` — the literal becomes
   a parameter
3. Tests: the defaults are unchanged, the parameter reaches `disk()`, and the **trade-off
   direction** is pinned as a property — a larger opening merges touching particles
4. The `5` floor gets a name too; it is the third undocumented number on that line

**Out of scope**

- **Changing either value.** The measurement says keep them
- **B-067** — an adaptive margin that tracks the largest particle
- **B-062** — `afm_sparse_low_snr` scores 0 at *every* factor in the sweep, which is more evidence
  that its problem is the detector's threshold and not the substrate

---

## Expected blast radius, before measuring

- **Zero golden differences.** The defaults are the current values and the arithmetic is
  unchanged; this is a naming and plumbing change with a measurement attached.
- If anything moves, a default was mistyped.

---

## Definition of done

- [x] Both factors and the floor are named constants with the sweep in their docstrings
- [x] `opening_scale` is a parameter; the default reproduces today exactly
- [x] Tests — **12**, including the trade-off property and the defaults asserted literally
- [x] `make check` green — 476 tests; delta **zero, golden byte-identical**
- [x] ADR-0037 carrying the full sweep; **B-067 filed**; `Backlog.md` (B-064 → done), `STATE.md`,
      `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [x] Commit: `M3-T26: the opening-radius constants are named, exposed and measured`

---

## What it turned up

**The flat response of the rough factor explains the previous task.** Recall and precision are
identical from 1.3 to 2.4 — so ADR-0035's truncation, which shifted that factor's *effective*
value to 1.39, could never have shown up in any quality metric. **A constant whose value does not
measurably matter does not get audited**, and that is a general lesson about where defects hide,
not a fact about this one.

**B-067 came out of the sweep rather than out of reading.** `afm_tilted_polydisperse` is the only
phantom that loses a detection as the factor grows, and it is the polydisperse one — which is the
signature of a margin derived from the *median* particle being too small for the large half of the
distribution.

---

## Notes

The outcome "the constants were already right" is a real result, not a wasted task: it is the
difference between a number nobody has checked and a number someone has. The sweep goes in the
ADR so the next person to change it starts from data.
