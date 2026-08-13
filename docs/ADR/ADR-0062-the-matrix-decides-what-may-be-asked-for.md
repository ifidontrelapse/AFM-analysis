# ADR-0062 — The matrix decides what may be asked for

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M6-T02)
- **Affects:** `application/capabilities`, `application/use_cases`, `gui/panels`, `gui/viewmodels` · M6

## Context

M6's third exit criterion names this panel:

> *Invalid combinations are disabled in the UI **because the capability matrix says so** — not by a
> duplicated rule.*

The matrix has existed since M2-T10 and has exactly one caller: `run_pipeline`, which validates a
request that has **already been assembled**. Nothing has ever asked it what to *offer*.

The difference is not cosmetic. A UI that cannot express an invalid request and a UI that lets one
be built and then apologises are the same screen until something goes wrong — and D-19 is what the
second one looks like after a year: the deleted React client kept its own copy of the matrix, and
the copy had drifted.

PROJECT_RULES §2.5 states the constraint the other way round: *the strings `"yolo"`, `"sam2"` and
`"log"` must not appear in `gui/`*. A widget that knows one detector's name is a widget that grows
an `if` about it.

## Decision

### 1. `detector_options(modality, frameworks, has_predictor)` answers "what may be offered"

In `application/capabilities.py`, beside the matrix it reads. It returns one entry per detector,
each carrying its modes **for that modality**, whether each can run, and — when it cannot — a
sentence saying why.

The panel renders that and decides nothing. Its combos contain the matrix's own strings, which it
received rather than typed.

### 2. An unavailable entry is offered, disabled, and explains itself

Three reasons exist today and none of them is the operator's fault:

- **the mode needs a predictor** — the matrix's own `requires_predictor`, and nothing constructs one
  before M6-T04;
- **the detector needs weights** — a framework with no model registered in this project (ADR-0050);
- **the mode is AFM-only** — `baseline` measures height above a substrate, so it needs a Z map, and
  for an SEM image the row simply does not exist.

The first two are *disabled and explained*; the third is **absent**, because the matrix has no such
row and inventing a greyed-out one would be the widget re-stating the rule.

**"Greyed out with no explanation" is the failure this criterion exists to prevent**, not a milder
version of it. `"you need to register a model"` and `"this application cannot do that"` are
different sentences, and an operator who cannot tell them apart files the second as a bug.

### 3. Running it stores a run — this is not a preview

`run_analysis` writes the run, its detections and its measurement table (ADR-0042). M6-T01's preview
was explicitly *not* a result; this explicitly is, which is why ADR-0061 §5 was worth writing down
one task earlier.

### 4. The preprocessing parameters travel with the run

`run_analysis` used to call `run_preprocessing` with its own defaults, so a scan previewed at
`opening_scale = 4.0` would have been **analysed at 2.5 with nothing saying so**. `PreprocessingParams`
is one value held by the session and written by the preprocessing panel on every change, and
`run_analysis` takes it.

One object rather than three keyword arguments threaded through three layers: the preview and the
run must use the same numbers, and the way they stay the same is that there is one value to hand
over.

### 5. Detection runs as a job, and cancelling it is honest

The third consumer of M5-T07's runner. It is also the first where ADR-0043 §3's limit bites: a LoG
pass over a 4096² scan has **no checkpoint**, so cancel is recorded and the pass finishes. The
button says *Stopping…* and means it, which is exactly what that ADR decided a cancel button is
allowed to promise.

## Consequences

**Positive**

- The matrix now has a second caller, and it is the one that decides what a person can ask for.
- Adding a row to `CAPABILITIES` changes the UI, with no widget edited — which is what "not a
  duplicated rule" has to mean to be worth anything.
- A run made from the window is a stored run, so M6-T09 has something to restore.
- The preview and the analysis cannot silently disagree about the substrate.

**Negative**

- `detector_options` knows which framework each detector needs (`DETECTOR_FRAMEWORKS`). That is one
  more place naming the detectors — in `application`, where they are already named, but it is a
  second table beside `CAPABILITIES` and the two must grow together. A single table with the
  framework on the row would be better and is a change to a structure the golden-adjacent validator
  reads; not taken in this task.
- Only the blob detector's parameters are on screen. The other detector is disabled in a fresh
  project, so the panel has never had to show its parameters — and when M6-T04 registers a model,
  this panel needs a second parameter group and a model picker.
- A run cannot be undone from here. Deleting a run is a repository operation with a confirmation
  behind it (ADR-0044's argument, one table over), and nothing asked for it yet.

**Neutral**

- `PipelineConfig` still carries `yolo_model_path`, so W10 stays *closable* rather than closed. The
  panel never sets it, because the detector that would use it cannot be selected without a
  registered model — and wiring the registry into `PipelineConfig` is a change to what
  `run_pipeline` reads, which ADR-0010 says gets its own commit.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Let the panel list detectors and modes | PROJECT_RULES §2.5, and D-19: the copy drifts |
| Hide unavailable entries | "Cannot do that" and "not set up yet" become the same silence |
| Grey out `baseline` for SEM instead of omitting it | The matrix has no such row; a greyed row would be the widget restating the rule |
| Validate only on Run | D-14 in a new place: an invalid request assembled, then apologised for |
| Keep `run_analysis` on its own preprocessing defaults | A scan previewed at one opening scale, analysed at another, silently |
| Three keyword arguments instead of `PreprocessingParams` | Three layers of threading, and three chances for one of them to be dropped |

## Compliance

- `tests/gui/test_detection_panel.py::TestTheOptionsAreTheMatrix` asserts the offered
  (detector, mode) pairs **are** `CAPABILITIES`' own rows for each modality, and that an SEM image is
  not offered the AFM-only mode.
- `TestAnUnavailableEntrySaysWhy` asserts a disabled entry carries its reason as a tooltip, that
  registering a model enables the detector that needed one, and that the combos open on something
  that can run.
- `TestNoDetectorNameLivesInTheGui` greps every module under `gui/` for the model names — with a
  test that proves the check can fail — and parses the panels for mode literals.
- `TestRunningIt` asserts a stored run with its detections, and that the preprocessing parameters
  reach it.

## References

- M2-T10 / `application/capabilities.py` — the matrix, and D-14/D-19 behind it
- ADR-0042 — what a run is, and where its measurement table goes
- ADR-0050 — the registry that decides whether a detector has weights
- ADR-0061 §5 — the preview/result distinction this task is the other half of
- ADR-0043 §3 — what the cancel button promises for a pass with no checkpoint
