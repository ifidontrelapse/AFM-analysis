# CURRENT TASK

**ID:** `M6-T02`
**Title:** A detection panel that offers what the matrix allows, and says why not
**Milestone:** M6 — Analysis workflow in the GUI, second task
**Defect:** — · **ADR:** **ADR-0062**
**Branch:** `feat/m6-analysis-workflow`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M6's third exit criterion is written against exactly this panel:

> *Invalid combinations are disabled in the UI **because the capability matrix says so** — not by a
> duplicated rule.*

The matrix has existed since M2-T10 and has one caller: `run_pipeline`, which validates a request
that has already been assembled. Nothing has ever *asked it what to offer*. That is the difference
between a UI that cannot express an invalid request and a UI that lets one be built and then
apologises — and D-19 is what the second one looks like after a year: the React client had a copy of
the matrix, and the copy had drifted.

It is also the first task that produces something the project **keeps**: `run_analysis` stores a run,
its detections and its measurement table (ADR-0042), and until now nothing in a window has called it.

---

## The decisions this task has to make

**1. Where do the detector and mode lists come from?** From the matrix, through `application`.

PROJECT_RULES §2.5: *the strings `"yolo"`, `"sam2"`, `"log"` must not appear in `gui/`*. So the panel
cannot enumerate detectors — it asks `application.capabilities` for the options for **this image's
modality** and renders what it is handed. A widget that knows one detector name is a widget that
will grow an `if` about it.

**2. What does "unavailable" mean, and who explains it?** The application, in a sentence.

Three reasons a row cannot run today, and none of them is the operator's fault:

- **`segment` needs a predictor** — the matrix says `requires_predictor`, and nothing constructs one
  yet (M6-T04);
- **a detector needs weights** — `yolo` needs an `ultralytics` model *registered in this project*
  (ADR-0050), and a fresh project has none;
- **`baseline` is AFM only** — the matrix's own row, which is also why an SEM image must not offer
  it.

Each disabled entry carries its reason where the operator will read it. **"Greyed out with no
explanation" is the failure mode this criterion exists to prevent**, not a lesser version of it.

**3. What does the run produce?** A stored run — this is not a preview.

`run_analysis` writes the run, its detections and `measurements.csv` (ADR-0042). M6-T01's preview was
explicitly *not* a result; this explicitly is, and the difference is the whole reason ADR-0061 §5 was
written down.

**4. Do M6-T01's preprocessing parameters reach it?** Yes, or the panel above it is a lie.

`run_analysis` calls `run_preprocessing` with its own defaults, so a scan previewed at
`opening_scale = 4.0` would be *analysed* at 2.5 and nothing would say so. The three parameters
become arguments there too, and the session passes what the preprocessing panel last used.

**5. Does it run as a job?** Yes — the third consumer of M5-T07's runner, and the first one where
cancelling matters, since a LoG pass over a 4096² scan has no checkpoint at all (ADR-0043 §3).

---

## Scope

**In scope**

1. `application/capabilities.py` — `detector_options(modality, frameworks, has_predictor)`, returning
   what may be offered and why the rest may not
2. `application/use_cases/analysis.py` — the preprocessing parameters, passed through
3. `gui/panels/detection.py` — detector, mode, the LoG parameters, Run, and what came back
4. `gui/viewmodels/session.py` — `detect(...)` as a job, and the stored run announced
5. **ADR-0062** — the matrix decides what is offered; an unavailable row explains itself
6. Tests: the options are the matrix's own rows, an SEM image is not offered `baseline`, a project
   with no model does not offer the detector that needs one, `segment` is refused with its reason,
   the run is stored with its detections, and the preprocessing parameters reach the run

**Out of scope**

- **Drawing detections on the canvas** — M6-T03, the next task
- **Segmentation** — M6-T04 builds the predictor whose absence this task reports
- **Choosing *which* registered model to use** — one framework has at most one usable model in a
  fresh project; the picker arrives with M6-T04's model management

---

## Definition of done

- [x] The panel offers exactly what `CAPABILITIES` allows for the image's modality
- [x] Every unavailable entry says why, in a sentence from `application`
- [x] No detector name appears anywhere under `gui/`
- [x] A run is stored, with its detections and its measurement table
- [x] ADR-0062 + the ADR index
- [x] `make check` green — 1074 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M6-T02: a detection panel that offers what the matrix allows, and says why not`

---

## What it turned up

**The name guard failed on this panel's own docstring**, which quoted PROJECT_RULES §2.5 —
including the two model names the rule is about. The rule's point is that a reader of `gui/` should
not learn those names, and a docstring teaching them is the same leak in prose, so the wording
changed rather than the guard. **A strict guard that occasionally forces a rephrase beats a clever
one with holes.**

**`run_analysis` preprocessed with its own defaults**, so a scan previewed at `opening_scale = 4.0`
would have been *analysed* at 2.5 with nothing saying so. Found by asking where the M6-T01 panel's
numbers actually went. `PreprocessingParams` is now one value the session holds and both paths read.

**`QComboBox.model()` is typed as the abstract base**, so disabling an entry — Qt's own way, through
`QStandardItemModel.item(i)` — does not type-check. One helper with a `cast` and a comment; the
alternative is writing a magic value into `UserRole - 1`, which is the same thing with the type
information thrown away.

**`detector_options` needed a second table.** Which framework a detector requires is not in
`CAPABILITIES`, so `DETECTOR_FRAMEWORKS` sits beside it and the two must grow together. Named as a
negative consequence in the ADR rather than hidden: putting the framework on the capability row is
better, and it changes a structure the request validator reads.
