# ADR-0032 — Scoring a detector against ground truth

- **Status:** Accepted
- **Date:** 2026-08-07
- **Affects:** `nanoscope/core/science/evaluation.py`, `tests/characterization/capture.py` ·
  M3-T15 · no audit defect — this is the **gap** five tasks wrote "not claimed" for
- **Numerical impact:** **7 golden differences, all of them `detection_quality: ADDED`.** Nothing
  moved; the harness gained the first block that records how *good* the detector is rather than
  what it did.

## Context

The characterization golden catches a number that moved. Nothing in the project said whether the
number was any *good*, and five tasks in this milestone hit that wall:

| Task | What it had to write |
|---|---|
| M3-T03 (D-03) | "Not claimed: better detections — the weights were trained on images the old path produced" |
| M3-T10 (D-12) | "Not claimed: better YOLO detections — inference is outside the gate" |
| M3-T21 (B7) | "No claim is made that either backend is better; **M3-T15 owns that**" |
| M3-T05 (D-09) | "Inventing a LoG confidence … **M3-T15 is the only thing that could license one**" |
| M3-T14 (D-17) | the same, one column further along |

Meanwhile `tests/characterization/phantoms.py` has carried exact ground truth — centres, radii,
heights — since the audit, and said so in its own docstring: *"Ground truth is returned alongside
the image so that a future evaluation harness can score detection against it."* The data to
answer the question has been in the repository the whole time, used only to detect change.

## Decision

### A match is a centre inside the particle

`distance(detection, particle) <= match_factor * particle_radius_px`, with `match_factor = 1.0`.

**Scale-free by construction**, which a fixed pixel threshold is not: the phantom set runs from
1.95 to 29.3 nm/px, so "within 3 px" is two different physical tolerances at the two ends of it,
and the same offset would be a hit on a large particle and a miss on a small one.

Not IoU of the two discs: a `Detection` is a centre and a radius, so an IoU here would be computed
between two *idealised* circles — a precision the inputs do not have. When there are real masks to
compare (segmentation), IoU is the right criterion and this is not that task.

### One detection per particle, assigned optimally

A detector that fires ten times on one particle scores one true positive and **nine false
positives**. That is what makes precision mean anything.

The assignment minimises total centre distance over the admissible pairs
(`scipy.optimize.linear_sum_assignment`) rather than taking nearest-first. Greedy gets the same
*counts* in most cases but can pick the wrong pairing, and then the localisation error is measured
between the wrong two objects — a test pins a case where greedy costs 6.0 and the optimum is 4.0.

### Ratios with a zero denominator are `None`

A detector that reported nothing on an empty image has no precision, and `1.0` would be a
substitute value — the seventh this milestone would have had to delete after ADR-0019, 0024,
0025, 0027, 0028 and 0031. `None` is also what the `_nm` fields are when there is no scale.

### It lives in `core/science/`, not in `tests/`

It is a library capability. Scoring a detector against annotated real images is what **M4** (the
project format, annotations) and **M8** (training) need, and a function under `tests/` cannot be
imported by either. Pure NumPy/SciPy, modality-neutral, no I/O — it belongs to the domain by every
rule in ADR-0001.

### The LoG scores go into the golden

Detection on a phantom is deterministic, so the numbers are stable. Recording them makes a
regression in detection **quality** visible next to the regressions in detection **values** the
golden already catches. YOLO's cannot be recorded: inference is outside the gate and CI has no
weights (PROJECT_RULES §6).

## Consequences

**Positive**

- "This change improved detection" becomes a sentence with evidence behind it, for the first time.
- A regression that keeps the pipeline running but finds fewer particles now fails the gate.
- The five deferred claims have somewhere to go. None of them is retroactively settled — see
  below — but the next one will be.
- M4 and M8 get the scoring function they were going to need anyway.

**Negative**

- The golden now depends on the LoG detector's output for seven phantoms in a second place, so a
  deliberate detector change updates two blocks instead of one. That is the cost of measuring
  quality at all.
- ~1 s added to the golden run.
- A number that looks like a verdict invites being read as one. The limits below are in the module
  docstring, not only here.

**Neutral**

- No existing behaviour changes. This commit adds a module and a harness block; nothing else moves.

## What this does not license

**A phantom is not a sample.** Scoring well on seven synthetic images licenses the sentence "this
change improved detection on the phantom set", and nothing about real scans. Real-sample
evaluation needs annotated real data, which is **B6 / M3-T16** and still waiting on the operator.

**These are baselines, not a before/after.** The "before" was never recorded and cannot be
recovered without re-running four superseded code paths. What is recorded is today's number,
against which the next change is measured.

**Segmentation is not scored.** The phantoms carry centres and radii, not masks.

## The measured delta

**7 differences, one per phantom, all `ADDED`.** No existing value moves — this commit adds a
module and a harness block and changes no behaviour.

### What the numbers say, the first time anyone has looked

| Phantom | TP | FP | FN | precision | recall | F1 | mean localisation | signed radius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `afm_flat_monodisperse` | 24 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.43 px · 0.86 nm | −0.43 px |
| `afm_tilted_polydisperse` | 30 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.61 px · 1.23 nm | −0.70 px |
| `afm_coarse_pixels` | 14 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.41 px · 4.05 nm | −0.19 px |
| `afm_dense_overlapping` | 59 | 1 | 11 | **0.983** | **0.843** | 0.908 | 0.83 px · 1.65 nm | −0.33 px |
| `afm_sparse_low_snr` | 0 | 0 | 6 | — | **0.000** | — | — | — |
| `sem_bright_particles` | 22 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.44 px · 0.66 nm | +0.19 px |
| `tem_dark_particles` | 22 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.36 px · 0.18 nm | +0.29 px |

Three things in that table are worth saying out loud.

**`tem_dark_particles`: 22 of 22, precision 1.000, 0.36 px.** ADR-0023 fixed D-12 and could only
report "0 → 22 blobs" — a count, from a detector nobody had scored. It is now a measurement: the
TEM path finds every particle, invents none, and lands a third of a pixel from the truth.

**`afm_sparse_low_snr`: recall 0.000.** The LoG detector finds **none** of its six particles at
the harness's settings. This is not new behaviour — M3-T12 already noticed the phantom produced
zero blobs on its ordinary path, and the golden had been recording a zero-column measurement table
for it since the baseline — but "0 blobs" and "**recall 0.0 against 6 known particles**" are
different sentences, and only the second one is a defect report. It is the first thing this
harness has found, and it belongs to whoever tunes the detector, not to this commit.

**Every AFM radius is biased small** (−0.19 to −0.70 px) and both image phantoms are biased large.
The sign is consistent within a modality, which is what a calibration offset looks like rather
than noise — and exactly the distinction the signed error was reported for.

**The dense case is the honest one.** 11 misses out of 70 where particles overlap, with one false
positive. That is the number any future change to the LoG parameters has to beat.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| A fixed pixel matching distance | Two different physical tolerances across a phantom set spanning 15× in scale | Every image had the same pixel size |
| IoU between detection disc and true disc | Computes a precision the inputs do not have — both discs are idealisations of a centre and a radius | The detector produced masks |
| Greedy nearest-first matching | Same counts, worse pairings, and the localisation error is then measured between the wrong two objects | Only the counts were reported |
| Report `precision = 1.0` when nothing was detected on an empty image | A substitute value in a metric, which is the one place a substitute value is guaranteed to be averaged into something | The metric were never aggregated |
| Put it in `tests/` next to the phantoms | It is needed by M4's annotation flow and M8's training loop, neither of which can import from the test tree | Scoring were only ever a test concern |
| Score YOLO too | Inference is outside the gate and there are no weights in CI. The function takes any detections; the *golden* can only record LoG's | CI could run the weights reproducibly |

## Compliance

- `tests/unit/test_evaluation.py` — **21 tests**: the perfect case, what counts as a match (and
  that the tolerance is the particle's own radius, not a pixel count), one detection per particle
  — ten boxes on one particle scoring 1 TP and 9 FP — the assignment being **optimal rather than
  greedy** on a constructed case where greedy costs 6.0 against the optimum's 4.0, the three empty
  cases and their `None` ratios, the signed-versus-absolute radius error distinguishing a
  calibration offset from scatter, the input checks, and one end-to-end run against the real LoG
  detector.
- Golden: `detection_quality` on all seven phantoms, recorded for the first time.

## References

- `tests/characterization/phantoms.py` — the ground truth, and its first-line promise
- `ADR-0019` / `ADR-0031` — absent is absent, applied here to ratios and to `_nm` fields
- `ADR-0023` (D-12, TEM) — the claim this harness can now score rather than infer
- **B6 / M3-T16** — real sample data, without which none of this extends to real scans
