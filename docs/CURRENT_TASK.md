# CURRENT TASK

**ID:** `M8-T08`
**Title:** Model evaluation report using the M3-T15 harness
**Milestone:** M8 — Training module, eighth and last task
**Defect:** — · **ADR:** **ADR-0088** (to be written)
**Branch:** `feat/m8-training`
**Status:** **planning 2026-09-03.** Not started.

---

## Why this task is last, and what it is for

All four of M8's exit criteria are met. This is the task that makes the milestone's own warning
actionable — the Roadmap states it against M8 and not against any single task:

> **Risk to scientific output:** new models change detections by design. Model comparison is
> reported through the M3 evaluation harness.

M8 now produces models (M8-T05), lets a project choose between them (M8-T06) and records which one
produced a run (M8-T06, schema v10). Nothing yet says **whether the new one is better**, and until
something does, *"new models change detections by design"* is a licence rather than a
control.

**And the harness has been waiting for this since M3-T15.** ADR-0032 put it in
`core/science/evaluation.py` and said why it was not in `tests/`: *"M4's annotation flow and M8's
training loop need it."* It closed on a sentence this task is the first to be able to answer —
**"a phantom is not a sample"**. M7 built the sample: hand-drawn boxes on real scans.

---

## What was measured before planning

**1. A stored run already says which model produced it.** M8-T06 added `analysis_runs.model_id` one
task ago, and it is what makes this report possible **without loading a single weight**: the project
keeps every detection a model ever made in this project, beside the annotations that are the truth.
An evaluation that re-ran inference would need ultralytics, would be out of the gate
(PROJECT_RULES §6), and would score a *different* run from the one an operator looked at.

**2. A model can be joined to the run that trained it, with no new storage.** M8-T04 registers a
model with `path = run.weights_path`, so `ModelDescriptor.path` and `TrainingRun.weights_path` are
the same string. The link exists today; nothing has used it.

**3. Which scans a model never saw is recoverable — until `cache/` is deleted.** Built with
`val_fraction=0.25` over eight annotated scans:

```
spec says:            6 train, 2 val
val stems on disk:    ['scan1', 'scan4']
train stems on disk:  ['scan0', 'scan2', 'scan3', 'scan5', 'scan6', 'scan7']
stems map back to image ids: True
held-out image ids:   [2, 5]
DatasetSpec fields:   ['root', 'classes', 'train_images', 'val_images']
```

So the **membership** lives only in the dataset directory, and the **counts** live on the spec.
That is not an oversight: ADR-0081 put datasets in `cache/` because they are re-creatable and
safely deletable, and M8-T01 put the counts on the spec *because* of it. The consequence is this
task's central honesty problem — **after `cache/` is deleted, nothing can say which scans a model
was trained on.**

**4. The harness scores centres and radii, and a box is neither.** `evaluate_detections` matches a
detection centre inside a truth radius. The conversion from a box already exists, in
`infrastructure/models/yolo.py`: centre is the box centre, radius is `min(w, h) / 2`.

---

## The decisions

**1. The report scores what the project already stored. It runs no model.**

For each image: the annotations are the truth, and the detections of a stored `AnalysisRun` are the
answer. Grouped by `run.model_id`, so two models are two columns over the same scans.

This is what makes it a *report* rather than a second analysis pipeline: no weights, no ultralytics,
no GPU, in the gate, and — the part that matters scientifically — it scores **the run the operator
actually looked at**, not a fresh one that might differ.

**2. Truth is the hand-drawn annotations, and the caller names the scope.**

`AnnotationSource.MANUAL` by default. ADR-0044's rule, third site: scoring a model against boxes
adopted from a detector is scoring it against a detector, and M7-T09 and M8-T05 both made the
caller say so out loud rather than defaulting into it.

**3. A score on a scan the model trained on is labelled, not hidden — and never silently averaged
with the rest.**

The report marks each image `unseen`, `trained-on` or `unknown`, from the training run found by
measurement 2 and the dataset directory from measurement 3. **`unknown` when `cache/` is gone**, and
that is stated rather than guessed: a model whose split cannot be recovered gets a score with the
provenance of that score attached, which is the same rule ADR-0025 applies to a missing pixel scale.

Totals are reported **separately for the unseen scans and for all of them**, because those two
numbers answer different questions and one of them is the only one that means *generalisation*.

**4. A box becomes a centre and a radius the same way on both sides.**

`min(w, h) / 2`, which is what a detector's own boxes already become. Using a different rule for the
truth would make the match tolerance and the radius error describe different circles — and the
matching rule is *a centre inside the particle*, so the radius **is** the tolerance.

**5. Comparison is over the same scans, or it is not a comparison.**

Two models scored on two different sets of scans produce two numbers that cannot be subtracted. The
report's unit is therefore *(model, image)*, and a summary row for a model states **how many images
it was scored on** beside its precision and recall — because that count is what makes the row
comparable or not.

**6. `precision` and `recall` stay `None` where the harness returns `None`.**

Not 0.0, not 1.0, not "—" computed into an average. ADR-0032 deleted the seventh substitute value in
this project for exactly this, and an aggregate that treats an absent ratio as zero is that value
coming back through a sum.

**7. The score lives where *compare* already lives.**

M8-T06 built the models dialog and wrote down what it was not doing: *"**Compare is the records**,
not a run of them — what a model does to a scan is M8-T08's report."* This is that task, so the
numbers go into that dialog rather than into a second window an operator has to know to open.

---

## Scope

**In scope**

1. `application/use_cases/evaluation.py` — the report: per model, per image, and two totals
2. Truth from annotations, scope named by the caller; `unseen` / `trained-on` / `unknown` per image
3. The models dialog shows each model's score, and says what it was scored on
4. **ADR-0088** + the ADR index
5. M8 closed: `Roadmap.md`, and the milestone summary in `Progress.md`

**Out of scope**

- **Running inference inside the report** — decision 1, and it would put the gate behind a GPU
- **Segmentation quality (mask IoU)** — the harness scores detection; ADR-0032 says so and the
  phantoms carried no masks either
- **Recording the held-out image ids** — measurement 3's gap. Named, and left to an operator's
  decision, because ADR-0081 chose deletability on purpose and reversing it needs more than this
- **A significance test** — two precisions on twelve scans is not a study, and dressing it as one is
  the claim ADR-0032 refused to license

---

## Definition of done

- [ ] Two models scored over the same scans, from what the project already stored
- [ ] A scan the model trained on is labelled, and the unseen total is reported separately
- [ ] A model whose dataset is gone reads `unknown`, never `unseen`
- [ ] An absent ratio stays absent, in the rows and in the totals
- [ ] ADR-0088 + the ADR index
- [ ] `make check` green, golden byte-identical
- [ ] Docs: `STATE.md`, `Progress.md` (+ the M8 summary), `TASKS.md`, `PROJECT_CONTEXT.md`, `Roadmap.md`
- [ ] Commit: `M8-T08: whether the new model is better, from what the project already kept`
