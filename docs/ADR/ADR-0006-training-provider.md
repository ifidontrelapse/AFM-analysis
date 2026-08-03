# ADR-0006 — Training is an application module behind `TrainingProvider`

- **Status:** Accepted
- **Date:** 2026-08-03
- **Affects:** `core/ports/training.py`, `infrastructure/training/`, `gui/views/training` · M8

## Context

The operator annotates their own scans, so their annotations should be able to become a
detection model without leaving the application. Nothing of this exists today: models are
consumed, never produced, and annotations have nowhere to go.

Training differs from inference in every operational dimension: it runs for hours instead
of seconds, it needs a dataset rather than an image, it produces artifacts and metrics
rather than detections, it must be cancellable and resumable, and it may run somewhere
else entirely — a lab workstation, a cluster, a rented GPU.

The tempting shortcut is to add a `train()` method next to `detect()` on the detector.
That is the wrong seam: it would put a multi-hour, artifact-producing, dataset-consuming
operation inside the object used for per-image inference.

## Decision

Training is a **separate application module** behind a `TrainingProvider` port.

- `core/ports/training.py` defines `TrainingProvider`: submit a training job from a
  dataset specification and a configuration; observe status, metrics and progress; cancel;
  collect artifacts.
- Two implementations, both satisfying the same port:
  - **`LocalTrainingProvider`** — trains on this machine (ultralytics), device resolved
    by the Device Manager (ADR-0004).
  - **`RemoteTrainingProvider`** — submits to a remote worker over a defined protocol.
- The **GUI talks only to the port.** It cannot tell local from remote, and must not try.
- Training consumes a dataset built from annotations (M8-T02) and produces a
  `ModelDescriptor` plus a persisted `TrainingRun` (config, metrics, artifacts,
  provenance, device).
- **Training is never invoked from the detection path.** No detector triggers training;
  no training code imports a detector's inference path.

## Consequences

**Positive**

- Remote training becomes a configuration choice, not a rewrite.
- The full loop — annotate, train, register, detect with the new model — closes inside
  one application, which is the main product differentiator over a pile of scripts.
- Training runs are reproducible records, not shell history.
- Long-running work sits behind the same job abstraction as everything else (M4-T06), so
  progress and cancellation are solved once.

**Negative**

- Two implementations of a port whose second implementation has no user yet; the remote
  protocol risks being designed for an imagined deployment.
- Streaming metrics out of ultralytics requires callback plumbing that is version-fragile.
- Cancellation and resumption of a multi-hour GPU job are genuinely hard to get right,
  especially remotely.
- Dataset building from annotations introduces a second annotation contract (the training
  format) that must stay in sync with the internal one.

**Neutral**

- Training pulls heavy optional dependencies; they load only when a training provider is
  constructed.

## Alternatives considered

| Alternative | Why not |
|---|---|
| `train()` on the detector interface | Conflates a per-image, second-scale operation with a multi-hour, artifact-producing one. Forces every detector — including LoG, which cannot be trained — to answer for a method it has no meaning for. |
| Training as an external script the user runs manually | Loses provenance, metrics, model registration and cancellation. It is the status quo, and it is why no model produced so far has a recorded lineage. |
| Local-only training | Simpler now, but the operator's laptop is not always the right GPU, and retrofitting remote execution into a synchronous local API is the expensive version of this decision. |
| Full experiment-tracking integration (MLflow, W&B) | External service or server for an offline desktop tool. The `TrainingRun` record covers the need; integration stays possible behind the same port. |

## Compliance

- `gui/` imports `TrainingProvider` from `core/ports/`, never a concrete provider.
- Both providers pass the same contract test suite.
- No module under `infrastructure/models/` (inference) imports
  `infrastructure/training/`, and vice versa.
- Every completed training run yields a persisted `TrainingRun` and a registered
  `ModelDescriptor`, or it is reported as failed — no silent artifacts on disk.

## References

- `systempromt.md` (Training)
- `docs/Architecture.md` §4.3
- `docs/TASKS.md` M8
- ADR-0004, ADR-0005
