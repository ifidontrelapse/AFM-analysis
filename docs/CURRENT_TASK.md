# CURRENT TASK

**ID:** `M3-T15`
**Title:** An evaluation harness — precision, recall and localisation against ground truth
**Milestone:** M3 — Numerical correctness, nineteenth task and the last of its numerical work
**Defect:** none. This is the **gap** five tasks have written "not claimed" for · **ADR:** **ADR-0032**
**Branch:** `sci/m3-numerical-correctness` (the consolidated branch — see the declared
deviation from PROJECT_RULES §7 in `STATE.md`)
**Status:** **done 2026-08-07.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

Every numerical defect the audit reproduced is closed. What is not closed is the project's
inability to say whether any of it *helped*:

| Task | What it had to write |
|---|---|
| M3-T03 (D-03) | "Not claimed: better detections — the weights were trained on images the old path produced" |
| M3-T10 (D-12) | "Not claimed: better YOLO detections — inference is outside the gate" |
| M3-T21 (B7) | "The backends are not bit-identical … **no claim is made that either is better**; M3-T15 owns that" |
| M3-T05 (D-09) | "Inventing a LoG confidence … **M3-T15 is the only thing that could license one**" |
| M3-T14 (D-17) | the same, one column further along |

Five tasks, one missing measurement. Until it exists, "the detector improved" is an opinion, and
the phantoms — which carry exact ground truth and have done since the audit — are being used only
to detect *change*, never *quality*.

`tests/characterization/phantoms.py` has said so from its first line: *"Ground truth is returned
alongside the image so that a future evaluation harness can score detection against it."*

---

## The decisions this task has to make

### 1. What counts as a match

| | |
|---|---|
| Centre within a fixed pixel distance | Simple, and wrong across phantoms: `afm_coarse_pixels` is 29.3 nm/px and `afm_flat_monodisperse` is 1.95, so one threshold means two different physical tolerances |
| **Centre within `match_factor × the particle's own radius`** ✅ | Scale-free by construction, and it states the criterion in the only unit that matters — "the detection landed on the particle" |
| IoU of the two circles above a threshold | The right criterion when both sides have real masks. Detections here are centre + radius, so an IoU would be computed between two idealised discs — precision the inputs do not have |

Default `match_factor = 1.0`: the detection's centre must fall inside the true particle.

### 2. One detection per particle, chosen optimally

A detector that reports ten boxes on one particle must be charged nine false positives, not
credited ten times. So: **one-to-one assignment**, over the admissible pairs only, minimising
total centre distance — `scipy.optimize.linear_sum_assignment`. Greedy nearest-first is easier
and can pick a worse global assignment; the cost of doing it properly is one scipy call.

### 3. What is reported

`TP / FP / FN`, `precision`, `recall`, `f1`, the localisation error of matched pairs (mean and
median, in pixels and — when a scale is known — in nanometres), and the **radius** error, because
the phantoms carry true radii and every downstream size statistic depends on them.

Absent stays absent (ADR-0019): with no `pixel_size_nm`, the nanometre fields are `None`, not the
pixel value wearing nanometre units.

### 4. Where it lives, and whether it is in the gate

`nanoscope/core/science/evaluation.py` — pure NumPy/SciPy, modality-neutral, no I/O. It is a
library capability, not a test helper: annotating real images and scoring a detector against them
is what M4 and M8 need, and a function living in `tests/` cannot be imported by either.

**And the LoG detector's scores go into the golden**, per AFM phantom. Detection on a phantom is
deterministic, so the numbers are stable; recording them makes a regression in detection *quality*
visible for the first time, next to the regressions in detection *values* the golden already
catches.

---

## Scope

**In scope**

1. `core/science/evaluation.py` — `match_detections`, `evaluate_detections`, and a
   `DetectionMetrics` dataclass with every field named for what it is
2. Validation at the entry, per ADR-0030
3. A `detection_quality` block in the harness: LoG against ground truth on all five AFM phantoms,
   plus the two image phantoms
4. Tests: perfect detection, duplicates, misses, a shifted set, scale-free matching, the empty
   cases, and the assignment being optimal rather than greedy

**Out of scope**

- **Any change to a detector.** This task measures; it does not tune. A commit that improves a
  number *and* the thing measuring it is a commit that cannot be read
- **YOLO's scores.** Inference is outside the gate (PROJECT_RULES §6) and there are no weights
  here or in CI. The function works on any detections; the golden can only record LoG's
- **Segmentation quality** (mask IoU against a true mask). The phantoms carry centres and radii,
  not masks; a mask ground truth is a phantom change and a task of its own
- **A verdict on M3.** The numbers this produces are today's, measured for the first time. They
  are a baseline, not a before/after — the "before" was never recorded and cannot be recovered
  without re-running four superseded code paths

---

## Definition of done

- [x] `evaluate_detections` returns precision, recall, F1, localisation and radius error, with
      the nanometre fields absent when the scale is
- [x] Matching is one-to-one and optimal, and a test pins a case where greedy costs 6.0 against
      the optimum's 4.0
- [x] The golden records LoG's scores on all five AFM phantoms **and both image phantoms**
- [x] Tests — **21**
- [x] `make check` green — **415 tests**; delta **7 differences, all `detection_quality: ADDED`**
- [x] ADR-0032; `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index, Backlog
- [x] Commit: `M3-T15: precision, recall and localisation against ground truth`

---

## What it turned up

**`afm_sparse_low_snr` scores recall 0.000** — six particles, none found. M3-T12 had already
noticed the phantom produces zero blobs, and the golden had been recording a zero-column
measurement table for it since the baseline; nobody read it as a defect, because "0 blobs" is a
number and "recall 0.0 against six known particles" is an accusation. **Filed as B-062.**

**`tem_dark_particles` closes a loop four days old.** ADR-0023 fixed D-12 and could only report
"0 → 22 blobs". It is now 22 of 22 with precision 1.000 and a localisation error of 0.36 px — the
count became a measurement.

**The radius error has a sign, and the sign is consistent.** Negative on every AFM phantom
(−0.19 to −0.70 px), positive on both image phantoms. That pattern is a calibration offset, not
scatter, and the mean *absolute* error alone could not have shown it — which is why both are
reported.

---

## Notes

What this can and cannot license, stated before the numbers exist so that the numbers cannot
quietly widen it: **a phantom is not a sample.** Scoring well on eight synthetic images licenses
the sentence "this change improved detection on the phantom set", and nothing about real scans —
which is **B6 / M3-T16**, still waiting on the operator.
