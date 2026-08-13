# ADR-0069 — What persists is the run, not the session

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M6-T09)
- **Affects:** `gui/viewmodels`, `gui/panels/measurements` · M6

## Context

M6's fourth exit criterion is *"results persist across application restart"*, and most of it was
already true by construction: `run_analysis` stores a run, its detections and its measurement table
(ADR-0042), and M6-T03 loads the newest run when an image is selected.

So this is a task about **proving it and finding what does not** — the shape M4-T09 had, where no
production code was written for autosave and what shipped was the proof plus the one missing piece.

## Decision

### 1. The proof ends the process's grip on the project

The container is closed and the repository connection with it; a **new** container and a **new**
window open the same directory and find the run, its detections, its measurement table, its overlay
and its statistics. Anything less proves a cache.

### 2. Older runs are reachable

Three analyses of one scan leave three rows, and the window could reach exactly one — the newest.
That is the criterion satisfied on a technicality: an operator who ran a sweep of three parameter
sets could see the last of them. The measurements panel now carries a **run selector**, and choosing
a run is `session.select_run`, which is the same one place the rest of the selection lives (ADR-0057).

### 3. A fresh selection shows the newest run

Remembering which run the operator was looking at, per image, across restarts is a project-scope
preference (ADR-0047) with a lifetime nobody has asked for. The newest is the one they just made.

### 4. What does not survive says so

A restored segmentation run has its detections and **no masks** — ADR-0042 did not persist them and
ADR-0064 kept them in memory. An empty overlay reads as *"segmentation found nothing"*, so the panel
says *"its masks were not stored and cannot be redrawn"* instead.

**Which modes make masks is answered by the matrix**, not by a literal in a widget: the panel asks
`capabilities.find(...).requires_predictor`. The first draft compared `run.mode == "segment"` and
**M6-T02's guard caught it** — which is what that guard is for.

M6-T01's preview does not survive either, and never claimed to: it is not a result (ADR-0061 §5).

## Consequences

**Positive** — the criterion is proven rather than asserted; a parameter sweep is readable after the
fact; the one thing that legitimately vanishes announces itself.

**Negative** — the run selector lists runs by id, mode and detector, not by *what was different about
them*. A sweep of three opening scales looks like three rows that differ only by id, because the
parameters a run used are not stored with it. That is a real gap, and it belongs to whichever task
decides a run should record its `PipelineConfig`.

**Neutral** — masks stay unpersisted. Reversing that is a format decision with a migration behind it,
and ADR-0042's reasoning — the weights that produce them are outside the gate — has not changed.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Assert persistence with a repository test only | Proves the storage, not that the application reads it |
| Remember the selected run across restarts | A preference with a lifetime nobody asked for |
| Show every run's detections at once | Two runs differ *by* their detections; drawing both asks which |
| Leave the overlay empty for a restored segmentation | Reads as "found nothing", which is a different statement |

## Compliance

`tests/gui/test_restart.py` closes the container, opens a new one **and a new window**, and finds the
run, its detections, its table, its overlay and its statistics; it also asserts every stored run is
offered after a restart, that an older one can be shown, and that a restored segmentation run reports
its missing masks.

## References

- ADR-0042 — what a run stores, and what it does not
- ADR-0064 — masks in memory only
- ADR-0061 §5 — the preview that is not a result
- ADR-0062 — the matrix answering "which modes make masks"
