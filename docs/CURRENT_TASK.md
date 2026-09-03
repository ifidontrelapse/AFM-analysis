# CURRENT TASK

**ID:** `M8-T06`
**Title:** Model management UI: import, register, activate, compare
**Milestone:** M8 — Training module, sixth task
**Defect:** **W10** (open since M4-T13) · **ADR:** **ADR-0086** (to be written)
**Branch:** `feat/m8-training`
**Status:** **planning 2026-09-03.** Not started.

---

## Why this task is sixth

M8-T05 closed the first two of M8's four exit criteria: a model is produced, recorded and
registered from the window. **The third is this task's, and it is the one that is measurably
false today:**

> *A trained model is selectable for detection in M6 **with no code change**.*

It is false because of a line M4-T13 named and did not close. `PipelineConfig.yolo_model_path`
still defaults to `"./checkpoints/best12x.pt"`, and M4-T13 said so in its own entry: **W10 is not
closed by this task, it is made closable** — with M5 named as the payer. M5 did not pay, M6 built
the detection panel on top of it, and M8-T05 has now made the project produce models that nothing
can select.

So this task is where **the model an operator registers becomes the model the detector loads.**

---

## What was measured before planning

**1. A registered detection model is not selectable.** A project with `particles-v1` registered,
driven through the real panel:

```
detector options:                 [('log', True), ('yolo', True)]
registered models:                ['particles-v1']
panel config yolo_model_path:     ./checkpoints/best12x.pt
does the panel offer to pick one? False
```

The capability half is already honest — `yolo` is *available* **because** an `ULTRALYTICS` model is
registered (`detector_options` checks exactly that). It is the last hop that is missing: the panel
builds a `PipelineConfig` and never touches `yolo_model_path`, so **every** yolo run in this
application loads the same hardcoded file.

**2. That file is resolved against the working directory.**

```
from the repo root:   ./checkpoints/best12x.pt -> exists: True
from anywhere else:   ./checkpoints/best12x.pt -> exists: False
```

`checkpoints/best12x.pt` is on this machine and untracked (M1 untracked the weights), which is
**why nobody has met this**: run from the repository root it silently works. The same project, the
same button, a different working directory — a different model, or none.

**3. The failure is a raw `FileNotFoundError`, and it arrives late.**

```
type:              builtins.FileNotFoundError
is a NanoscopeError: False
message:           [Errno 2] No such file or directory: 'checkpoints/best12x.pt'
```

PROJECT_RULES §3 forbids that. And it is raised by `YOLO(self.model_path)` **inside**
`_detect_direct` — constructing a `YoloDetector` with a missing file raises nothing at all — so an
operator waits for a scan to be preprocessed and then gets a bare Python traceback naming a path
they never chose.

**4. `Settings` has a project scope that nothing writes.** `Settings.set(key, value,
Scope.PROJECT)` has existed since M4-T10, and M5-T09's dialog says why it offered nothing:
*"the project scope is not offered because this application writes no project-scoped setting yet."*

---

## The decisions

**1. `yolo_model_path` stops being a default that lies.**

Its default becomes `""`, and a `yolo` run with no weights named is **refused with a sentence**
before a detector is constructed — which is where D-14's rule already put this class of refusal
(M2-T10: *an impossible request refuses in milliseconds rather than after a GPU pass*). A path that
depends on the working directory is not a default, it is a guess about where the process was
started, and ADR-0025's rule applies unchanged: **an unknown is a state, not a value to invent.**

**2. `model_id` travels as an argument to `run_analysis`, not as a field on `PipelineConfig`.**

The caller names a model the way an operator does — by the id they gave it (ADR-0050) — and
`run_analysis` turns it into a path through `repository.path_of_model`, because resolving a
project-relative path is the repository's job and doing it anywhere else is the rule ADR-0038's
compliance section names. `run_pipeline` keeps taking a path, so a caller with weights and no
project — the golden, a notebook — is unaffected.

**An argument, because a field would move the golden.** `pipeline.py`'s own docstring says it:
*"the golden records the sorted field names of both classes, so adding or renaming one here is
drift"* — and `config_fields` in `baseline.json` is that list. Changing a **default** does not move
it; adding a field does, and the Roadmap's third sequencing rule is that *a golden update and a use
case never share a commit*. `run_analysis` already takes `predictor` and `preprocessing` outside the
config for the same practical reason, so this is the shape that is already here rather than a new
one.

**3. Which model a project detects with is a project-scoped setting.**

`models.active` through `Settings.set(..., Scope.PROJECT)` — the first writer of a scope that has
waited since M4-T10, and it is the right one by ADR-0047's own test: **a chosen model belongs to
the project, not to the person.** An operator with two projects has two answers, and putting this
in the application scope would leak one project's model into every other.

**4. A run records which model produced it.** Schema **v10**, one column.

Without it, *"which model found these particles?"* is unanswerable the moment a second model is
registered — and it is the question M8-T08's comparison is made of. It is also the same argument
M8-T04 made for training runs: the weights are on disk and the provenance is nowhere. `NULL` for
every existing row and for every `log` run, which is honest: those runs used no model.

**5. Importing weights registers them; it does not copy them.**

ADR-0050 already decided this and stated the consequence — an absolute path to a 137 MB checkpoint
is kept as it is, and *the project opens elsewhere with that model unavailable*. A dialog that
quietly copied gigabytes into `models/` would be making a storage decision on an operator's behalf;
one that refuses external weights would force it. The import asks for the id, the task and the
framework — **the three things a `.pt` file does not say** — which is `ImportOptions`' shape from
M5-T07 and `LabelSource`'s from M7-T09, for the third time.

**6. *Compare* is the records, not a run of them.**

Side by side: what each model was trained on, on how many images, its input size, its classes, when
it was registered, and whether the file is still there. **Not a score** — comparing models by
running them is M8-T08's evaluation report through the M3-T15 harness, and a comparison this task
invented would be the second answer to that question.

**7. A model whose file is gone is shown, not hidden.**

ADR-0040's rule, met from the model side: the row is real and the file is not, which is exactly what
`check_integrity` reports for images. Hiding it turns *"that model is on the other machine"* into
*"that model never existed"*, and the active one being missing is a refusal an operator has to be
able to see the reason for.

---

## Scope

**In scope**

1. `PipelineConfig.yolo_model_path` defaults to `""`; `run_pipeline` refuses a yolo run with no
   weights **before** constructing a detector — **W10 closed**, and the golden does not move
2. `run_analysis(..., model_id=...)` resolves the id → the weights path through the repository
3. Schema **v10**: `analysis_runs.model_id`; `schema_history.py` extended
4. `models.active`, project-scoped, and the session methods that read and write it
5. `nanoscope/gui/panels/models.py` (or a dialog): list, import, activate, compare, and what is
   missing
6. The detection panel uses the active model, and says so when there is none
7. **ADR-0086** + the ADR index

**Out of scope**

- **Running a comparison** — M8-T08's evaluation report, and decision 6
- **Copying weights into the project** — ADR-0050 decided it and stated the consequence
- **Deleting weights from disk** — a 137 MB artifact is an operator's decision; unregistering is not
- **Segmentation model selection** — M6-T04 takes the first `SEGMENT` model, and changing that rule
  is a second activation with no second caller asking for it yet

---

## Definition of done

- [ ] A model trained in M8-T05 is selected for detection **with no code change** — M8's third criterion
- [ ] No path in this application resolves against the working directory
- [ ] A yolo run with no model refuses with a sentence, before any file is read
- [ ] A stored run says which model produced it
- [ ] A model whose weights are missing is listed as missing, and cannot be activated silently
- [ ] ADR-0086 + the ADR index
- [ ] `make check` green, golden byte-identical
- [ ] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, `Backlog.md` (W10 closed)
- [ ] Commit: `M8-T06: the model an operator registers is the model the detector loads`
