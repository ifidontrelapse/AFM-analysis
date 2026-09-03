# ADR-0088 — A score is a score *of* something, and the report says what

- **Status:** Accepted
- **Date:** 2026-09-03
- **Deciders:** operator + agent (M8-T08)
- **Affects:** `application/use_cases/evaluation.py`, `gui/dialogs/models.py` · M8

## Context

The Roadmap states M8's risk against the milestone rather than any one task:

> **Risk to scientific output:** new models change detections by design. Model comparison is
> reported through the M3 evaluation harness.

M8-T05 produces models, M8-T06 lets a project choose between them and records which one produced a
run. Nothing said **whether the new one is better**, and until something does, *"new models change
detections by design"* is a licence rather than a control.

**The harness has been waiting since M3-T15.** ADR-0032 put it in `core/science/evaluation.py` and
said why it was not in `tests/`: *"M4's annotation flow and M8's training loop need it."* It closed
on a sentence that this task is the first thing able to answer — **a phantom is not a sample.** M7
built the sample: hand-drawn boxes on real scans.

Four things were measured before anything was written.

**1. A stored run already says which model produced it.** M8-T06 added `analysis_runs.model_id` one
task ago. The project therefore already holds every detection each model ever made in it, beside the
annotations that are the truth.

**2. A model can be joined to the run that trained it, with no new storage.** M8-T04 registers a
model with `path = run.weights_path`, so `ModelDescriptor.path` and `TrainingRun.weights_path` are
the same string. The link has existed since that task and nothing had used it.

**3. Which scans a model never saw is recoverable — until `cache/` is deleted.** Built with
`val_fraction=0.25` over eight annotated scans:

```
spec says:                   6 train, 2 val
val stems on disk:           ['scan1', 'scan4']
stems map back to image ids: True
held-out image ids:          [2, 5]
DatasetSpec fields:          ['root', 'classes', 'train_images', 'val_images']
```

The **counts** live on the spec for ever; the **membership** lives only in a directory ADR-0081
declared safely deletable, and M8-T01 put the counts there *because* of it. This is the report's
central honesty problem.

**4. A box is not a centre and a radius.** The harness matches a detection centre inside a truth
radius; the conversion already exists in `infrastructure/models/yolo.py` — centre is the box centre,
radius is `min(w, h) / 2`.

## Decision

### 1. The report scores what the project already stored, and runs no model

For each scan: the annotations are the truth and the detections of a stored `AnalysisRun` are the
answer, grouped by `run.model_id`.

Re-running inference here would need ultralytics, would put the gate behind a GPU (PROJECT_RULES
§6), and — the part that matters scientifically — would score a **different run** from the one the
operator looked at. It also makes M8-T06's new column load-bearing rather than speculative.

### 2. Truth is the hand-drawn boxes, and the caller widens the scope out loud

`AnnotationSource.MANUAL` by default. ADR-0044's rule at its third site: scoring a model against
boxes adopted from a detector is scoring it against a detector, and M7-T09 and M8-T05 both made the
caller say so rather than defaulting into it.

### 3. Exposure is a column, `unknown` is an answer, and the two totals are reported apart

Every row says whether the model was `unseen`, `trained-on` or `unknown` on that scan. Two totals
per model: **over the scans it never saw**, and over all of them. Only the first says anything about
generalisation, and reporting one number would make the reader guess which.

**`unknown`, not `unseen`, when the dataset directory is gone** — measurement 3, and the same rule
ADR-0025 applies to a missing pixel scale: an unknown is a state, and calling those scans unseen
would invent the one fact this report exists to be careful about. A model this project never trained
(an imported checkpoint, M8-T06) is `unknown` for the same reason.

### 4. The same box→circle rule on both sides

`min(w, h) / 2`, which is what a detector's own boxes already become. The harness matches *a centre
inside the particle*, so the truth radius **is** the tolerance: a circumscribed truth against
inscribed detections would compare two different circles and report the difference as a localisation
error.

### 5. Totals sum the counts and recompute the ratios. They never average ratios

A mean of per-image precisions weights a scan with two particles the same as one with two hundred,
and it has no denominator to be honest about — a scan the harness scored `None` would have to become
a number to be included. Summing the counts keeps the ratio's meaning and keeps `None` where the
denominator really is zero.

Localisation errors are averaged **weighted by true positives**, which is what one call over all
pairs would have produced. **Medians are not aggregated at all**: the median of a set of medians is
not a median, so the totals report `None` rather than a number that would read as one. The per-image
rows keep theirs, where it means what it says.

### 6. An absent ratio stays absent, on screen as well as in the record

Blank, not `0.000`. ADR-0032 deleted the seventh substitute value in this project for exactly this:
a detector that reported nothing has no precision, and a zero there is a measurement it never made.

### 7. The score goes where *compare* already is

M8-T06 built the models dialog and wrote down what it was not doing: *"Compare is the records, not a
run of them — what a model does to a scan is M8-T08's report."* This is that half, in the same
window, so an operator does not have to know a second one exists.

### 8. No significance test, and no claim beyond the scans

Two precisions over twelve scans is not a study. ADR-0032 refused to license more than *"this change
improved detection on the phantom set"*, and the same restraint applies with real scans: this says
what a model scored **on these scans, against these boxes**, and the report carries what it was
scored on so that the sentence can be written correctly.

## Consequences

**Positive**

- **M8's stated risk has a control.** *"New models change detections by design"* is now a sentence
  an operator can check, on their own data, in the window where they choose a model.
- **M3-T15's harness has a real sample at last**, five milestones after ADR-0032 said it would.
- The evaluation is **fast and in the gate**: no weights, no torch, no GPU. It is a few queries.
- Three things stored for unrelated reasons — M7's annotations, M6's runs, M8-T04's training record
  — turn out to join without a schema change. M8-T06's `model_id` column becomes load-bearing.
- A model that looks perfect on the scans it trained on and fails on the ones it did not is now
  **visible as exactly that**, which is the failure this milestone could otherwise have shipped.

**Negative**

- **`unknown` will become the common case.** `cache/` is deletable by design, and once it is gone
  this project can never again say which scans a model saw. The gap is named rather than closed:
  recording the held-out ids is a schema change reversing a decision ADR-0081 made on purpose, and
  no operator has yet asked for it.
- **A model is scored only where it has been run.** Two models compared over different scans produce
  numbers that cannot be subtracted; the report says what each was scored on, and does not stop a
  reader subtracting them anyway.
- **The newest run per model per scan wins**, so an older, better run is invisible. That is what
  re-running a model means, but it is a choice.
- **Detection only.** Segmentation quality (mask IoU) is not scored, because the harness does not —
  ADR-0032's own limit, unchanged.
- The dialog is now a records table, a score table and an import form in one window.

**Neutral**

- The exposure join costs one directory listing per model, once per open.
- Nothing new is stored. The report is derived, and re-derived every time it is shown.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Re-run each model inside the report | Needs ultralytics and a GPU, leaves the gate, and scores a different run from the one the operator saw |
| One total per model | Hides whether the number is about scans the model trained on; the two answer different questions |
| Call an unrecoverable split `unseen` | Invents the one fact the report exists to be careful about (ADR-0025's rule) |
| Record the held-out image ids in the schema | Reverses ADR-0081's deliberate deletability for a need no operator has stated; named as the gap instead |
| Average per-image precisions | Weights a two-particle scan like a two-hundred-particle one, and forces `None` to become a number |
| Report a median of medians | It is not a median |
| Show `0.000` for an absent ratio | The substitute value ADR-0032 deleted, arriving through a table cell |
| Count adopted boxes as truth by default | Scoring a model against a detector — ADR-0044, third site |
| Score every run rather than the newest per model | Counts one model twice on one scan and weights it by how often it was run |
| A significance test on the difference | Twelve scans is not a study, and ADR-0032 refused to license more than the phantom set |
| A separate report window | M8-T06 already built where *compare* lives and said this belonged there |

## Compliance

- `tests/integration/test_model_evaluation.py` — 15 tests: a perfect detector scores 1.0 and a
  half-blind one 0.5, from stored runs with **no weights loaded**; two models over the same scans; a
  run naming no model is attributed to none; re-running counts once; a scan the model trained on is
  labelled and kept out of the unseen total, whose recall is worse than the overall; a deleted
  dataset and an imported checkpoint both read `unknown` with `unseen is None`; adopted boxes are
  not truth by default and are once the caller says so; a scan with no annotations is not scored; a
  box becomes `min(w, h) / 2` at `[y, x]`; an absent precision stays absent in a row and in a total;
  totals sum counts; localisation is a mean over pairs, not over scans; a median of medians is not
  reported.
- `tests/gui/test_model_management.py` — the score table appears from a stored run, a miss reads as
  one with a blank precision, a model nobody has run is not a model that scored badly, and the
  window says when a split is no longer known.
- `tests/unit/test_import_graph.py` — caught `gui/dialogs/models.py` importing `core.science` for a
  type annotation on the first run; the widget reads fields and does not name the type.
- The golden is byte-identical. The harness is called, not modified.

## References

- ADR-0032 / M3-T15 — the harness, its one-to-one optimal matching, its `None` ratios, and *a phantom is not a sample*
- ADR-0044 — *a model trained on its own output is confirming itself*; §2 is its third site
- ADR-0081 — datasets in `cache/`, deletable by design, which §3 is the cost of
- ADR-0084 / ADR-0086 — the training record and `analysis_runs.model_id`, the two joins this report is made of
- ADR-0025 — an unknown is a state, not a value to invent
