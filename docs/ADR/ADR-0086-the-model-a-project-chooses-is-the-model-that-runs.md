# ADR-0086 — The model a project chooses is the model that runs

- **Status:** Accepted
- **Date:** 2026-09-03
- **Deciders:** operator + agent (M8-T06)
- **Affects:** `core/entities/pipeline.py`, `application/use_cases/analysis.py`,
  `application/settings.py`, `gui/dialogs/models.py`, `gui/panels/detection.py`,
  `infrastructure/storage/` · M8

## Context

M8-T05 closed M8's first two exit criteria. The third is this task's, and it was measurably false:

> *A trained model is selectable for detection in M6 **with no code change**.*

It was false because of **W10**, a weakness M4-T13 named, closed halfway, and handed on. Its own
entry says so: *"W10 is not closed by this task, it is made closable"* — with M5 named as the payer.
M5 did not pay it, M6 built the detection panel on top of it, and M8-T05 made the project produce
models that nothing could select.

Four things were measured before anything was written.

**1. A registered detection model was not selectable.** A project with `particles-v1` registered,
driven through the real panel:

```
detector options:                 [('log', True), ('yolo', True)]
registered models:                ['particles-v1']
panel config yolo_model_path:     ./checkpoints/best12x.pt
does the panel offer to pick one? False
```

The capability half was already honest — the framework-backed detector is *available* **because** a
matching model is registered, which `detector_options` checks. It was the last hop that was missing:
the panel built a `PipelineConfig` and never touched the weights path, so **every** such run in this
application loaded one hardcoded file.

**2. That file was resolved against the working directory.**

```
from the repo root:   ./checkpoints/best12x.pt -> exists: True
from anywhere else:   ./checkpoints/best12x.pt -> exists: False
```

The checkpoint is on this machine and untracked (M1 untracked the weights), which is **why nobody
had met this**: run from the repository root it silently worked.

**3. The failure was a raw `FileNotFoundError`, and it arrived late.**

```
type:                builtins.FileNotFoundError
is a NanoscopeError: False
message:             [Errno 2] No such file or directory: 'checkpoints/best12x.pt'
```

PROJECT_RULES §3 forbids that. Constructing the detector with a missing file raised **nothing**; the
error came out of the framework's loader *during* detection, so an operator waited for a scan to be
preprocessed and then got a bare traceback naming a path they never chose.

**4. `Settings` had a project scope that nothing wrote.** Offered since M4-T10, and M5-T09's dialog
says why it showed nothing: *"the project scope is not offered because this application writes no
project-scoped setting yet."*

## Decision

### 1. The weights path stops being a default that lies

`PipelineConfig.yolo_model_path` defaults to `""`, and a run that needs weights and names none is
**refused with a sentence before a detector is constructed** — where M2-T10 put this class of
refusal for D-14's reason: *an impossible request should refuse in milliseconds rather than after a
GPU pass.*

A path that resolves against the directory the process happened to start in is not a default; it is
a guess about the environment. ADR-0025's rule applies unchanged: **an unknown is a state, not a
value to invent.**

### 2. `model_id` travels as an argument to `run_analysis`, not as a field on `PipelineConfig`

The caller names a model the way an operator does — by the id they gave it (ADR-0050) — and
`run_analysis` resolves it through `repository.path_of_model`, because joining a project root to a
relative path is the repository's job and doing it anywhere else is what ADR-0038's compliance
section rules out by name. `run_pipeline` still takes a path, so a caller with weights and no
project — the golden, a notebook — is untouched.

**An argument, because a field would have moved the golden.** `pipeline.py`'s own docstring says it:
*"the golden records the sorted field names of both classes, so adding or renaming one here is
drift"*, and `config_fields` in `baseline.json` is that list. Changing a **default** does not move
it; adding a field does — and the Roadmap's third sequencing rule is that *a golden update and a use
case never share a commit*. `predictor` and `preprocessing` already travel outside the config for
the same practical reason, so this is the shape that was already here.

### 3. Which model a project detects with is a project-scoped setting

`models.active`, written through `Settings.set(..., Scope.PROJECT)` — **the first writer of that
scope**, which has waited since M4-T10. It is the right one by ADR-0047's own test: a chosen model
belongs to the project, not to the person. An operator with two projects has two answers, and the
application scope would put one project's choice in front of every other.

Activating an id this project does not have is **refused rather than stored**: a stored id nothing
resolves is a detection that fails later, for a reason nobody can see.

### 4. A run records which model produced it — schema v10, one column

`analysis_runs.model_id`, nullable, and `NULL` for every `log` run and every row written before
today. That is honest rather than lossy: those runs used no model.

It matters now because it did not before. With one hardcoded path there was nothing to record; with
a project able to hold three models and choose between them, *"which model found these particles?"*
is otherwise unanswerable — and it is the question M8-T08's comparison is made of. The same argument
ADR-0084 made one table over, for a training run's provenance.

The **id**, not the path: a path is where the weights were on the machine that ran it, and the id is
what this project calls the model.

### 5. Registering weights does not copy them

ADR-0050 decided this and stated the consequence in the same breath — an absolute path to a 137 MB
checkpoint is kept as it is, and *the project opens elsewhere with that model unavailable*. Copying
gigabytes into `models/` on an operator's behalf is a storage decision this layer does not get to
make; refusing external weights would force it.

The dialog asks for the id, the task and the framework, because **a weights file says none of the
three** — `ImportOptions`' shape since M5-T07 and `LabelSource`'s since M7-T09, and the third time
the same answer is right. Re-registering an id replaces it, which is what retraining means.

### 6. *Compare* is the records, not a run of them

The table **is** the comparison: what each model is called, whether it is in use, what it does, its
input size, the classes it can name, when it was registered, whether the file is there, and where it
came from. Every column is something a reader asks before choosing between two models.

**No score.** What a model *does* to a scan is M8-T08's evaluation report through the M3-T15
harness, and a second answer to that question invented here would be the copy that drifts — which is
D-19's whole lesson, and the reason PROJECT_RULES §2.5 is enforced by a grep rather than by review.

### 7. A model whose weights are gone is shown, not hidden

ADR-0040's dangling row, met from the model side: the row is real and the file is not. Hiding it
turns *"that model is on the other machine"* into *"that model never existed"*, and a run naming it
is refused with a sentence naming the path — not a framework's `FileNotFoundError` halfway through.

### 8. Registered is not chosen, and the panel says which is missing

The matrix refuses a detector whose framework has **no registered model**. A project can have three
registered and none in use, and without a second check that run is accepted, preprocesses a scan and
*then* refuses. So the panel disables Run and **names the menu that fixes it** — because "greyed out
with no explanation" is the failure M6's third exit criterion exists to prevent, and *"choose a
model"* without saying where is that failure with more words.

## Consequences

**Positive**

- **M8's third exit criterion is met, and W10 is closed** — nine months after M4-T13 named it, and
  by the task that finally had a caller for it.
- No path in this application resolves against the working directory. The same project behaves the
  same way from any shell.
- A run that cannot work refuses in milliseconds, with a sentence, instead of after a preprocessing
  pass with a framework's traceback.
- The project scope is no longer a feature with no writer, and the first thing put in it is the one
  that most obviously belongs to a project rather than a person.
- A stored run says which model produced its detections, which is what M8-T08 needs to compare two.

**Negative**

- **Two fields for one idea.** `model_id` is what was asked for and `yolo_model_path` is what the
  pipeline loads, and a caller can still set the second directly. That is deliberate — it is what
  keeps `run_pipeline` usable without a project — but it is a pair that can disagree if somebody
  sets both.
- **A project upgraded from v9 has `NULL` for every existing run.** Truthful, and it means *"which
  model produced this?"* stays unanswerable for everything analysed before today.
- **`checkpoints/best12x.pt` stops being picked up silently.** Anyone who relied on running from the
  repository root now has to register that file once. That is the point, and it is still a change in
  behaviour for the one machine where it worked.
- The models dialog is a table with nine columns. It is the comparison, and it is wide.

**Neutral**

- Schema **v10**, and the third migration in this project to add a column to an existing table.
- The dialog is modal, unlike M8-T05's training window: this one asks a question and takes an
  answer, where that one watches six hours of work.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Keep a default weights path, but make it project-relative | Still a file nobody promises exists, and still a guess — the fix for an unknown is to say so (ADR-0025) |
| `model_id` as a field on `PipelineConfig` | Moves the golden's `config_fields`, and the Roadmap's third sequencing rule keeps that out of a use case's commit |
| Resolve the path in `run_pipeline` | It has no repository, and giving it one puts project storage in the function the golden covers |
| Store the active model in the application scope | Leaks one project's model into every other — ADR-0047's first named failure mode |
| Store the resolved **path** on the run | It is where the weights were on the machine that ran it; the project's answer is a name (ADR-0050) |
| Copy imported weights into `models/` | Duplicates gigabytes per project, and ADR-0050 already chose the other way and stated its cost |
| Hide models whose weights are missing | Turns *"it is on the other machine"* into *"it never existed"*, and leaves an activation refusal with no visible cause |
| Let the panel run and fail later when no model is chosen | The late failure this whole task exists to remove |
| Score the models in the compare table | M8-T08's report through the M3-T15 harness; a second answer here is the copy that drifts (D-19) |
| Leave W10 for M9 | M8-T05 made the project produce models, and the third exit criterion is this milestone's |

## Compliance

- `tests/gui/test_model_management.py` — 22 tests: the weights a run loads are the active model's,
  resolved through the repository; no default path resolves against the working directory; a run
  with no weights refuses **before reading anything**; a `log` run records `NULL` and a model-backed
  run records the id, through a close and reopen; the choice is stored in the **project** scope and
  not the operator's, survives reopening, and is refused for an id the project does not have;
  weights are registered where they are and never copied, an absent file is refused, and a checksum
  is computed; a model whose weights are gone is listed as `missing` and a run naming it raises
  `MissingFileError`; the table is the comparison and marks the active row; and the detection panel
  disables Run with a sentence when a model is registered but none is chosen, re-enabling it without
  a restart.
- `tests/integration/schema_history.py` — v10's column is in the revert map, which its own guard
  test enforces.
- `tests/gui/test_detection_panel.py` — the §2.5 grep still passes, this task's two files included.
- The golden is byte-identical: only a **default** changed, and `config_fields` records names.

## References

- ADR-0005 / ADR-0050 — what a model is, who names it, and why weights are not copied
- ADR-0047 — two scopes, and how to choose between them; §3 is its first project-scope writer
- ADR-0038 / ADR-0042 — who joins a project root to a relative path, and where a run is stored
- ADR-0040 — the dangling row, and the obligation to report it; §7 is the model side
- ADR-0025 — an unknown is a state, not a value to invent; §1
- ADR-0062 — the capability matrix, and a UI that cannot express an invalid request; §8
- ADR-0084 / ADR-0085 — the run record and the window that produces the models this one manages
