# CURRENT TASK

**ID:** `M8-T01`
**Title:** The `TrainingProvider` port: what a training run is, before anything trains
**Milestone:** M8 — Training module, first task
**Defect:** — · **ADR:** **ADR-0080** (to be written)
**Branch:** `feat/m8-training` (to be created from the M7 close)
**Status:** **planning 2026-08-17.** Not started.

---

## Why this task is first

ADR-0006 chose the seam in M0 and said why: training runs for hours instead of seconds, consumes a
dataset rather than an image, produces artifacts and metrics rather than detections, must be
cancellable, and may run on another machine. *"The tempting shortcut is to add a `train()` method next
to `detect()` on the detector. That is the wrong seam."*

Everything else in M8 speaks this port's vocabulary: M8-T03 implements it on ultralytics, M8-T04
persists what it produced, M8-T05 puts its metrics on screen, M8-T07 satisfies it from another
machine. **Written after them, it is whatever the first implementation happened to need.**

M7-T09 built the input side — annotations leave as labels the trainer reads. Nothing in this project
produces a model.

---

## The decisions this task has to make

**1. A port with no implementation is a hypothesis, and this project has checked one before.**

M2-T08 wrote seven ports and implemented one, and recorded that as a decision so the absence would
read as intent later. ADR-0041 sharpened it: *a use case earns its place or is not written*. So the
surface here is **only what M8-T02…T05 will actually call**, and the deliverable that keeps it honest
is the **contract test suite both providers must pass** (ADR-0006's own compliance clause) — run in
this task against a fake provider written for it, and against `LocalTrainingProvider` in M8-T03.

**2. Is a training run a `Job` (ADR-0043)?**

The runner already does submit, progress, cooperative cancellation and *"the listener fires on the
worker thread"*. But a `Job` is **in-process and dies with the process**, and a training run is hours
long, has to be findable after a restart (M8-T04), and may be executing on a machine this application
did not start. The likely answer — to be argued in ADR-0080 — is that the **port has its own handle
and status**, and the *local* provider drives it with the `JobRunner` underneath; a remote one polls.
What must not happen is a second thread policy in the layer ADR-0043 already settled.

**3. What is a metric?**

Trainers report different numbers. This is exactly the shape ADR-0031 met (four producers, four
schemas) and answered with *a core plus blocks*, and ADR-0042 met again and answered with *a file*.
The candidate: an epoch number and a **mapping of named scalars**, declared once, with the same rule
ADR-0031 wrote — one quantity, one name, and a block that is present in full or absent in full.

**4. Cancellation means what ADR-0043 made it mean.**

*Stop at the next checkpoint*, and for a trainer the checkpoint is an epoch boundary — which the GUI
must say, or it is a button that appears to do nothing (M5-T07 already learned this once).

**5. Artifacts are files, and a finished run registers a model.**

PROJECT_RULES §5 puts weights under `models/`; ADR-0050's registry hands back **factories, never
instances**, and takes a path. ADR-0006's compliance clause: *every completed training run yields a
persisted `TrainingRun` and a registered `ModelDescriptor`, or it is reported as failed — no silent
artifacts on disk.* This task declares that contract; M8-T04 persists it.

---

## Scope

**In scope**

1. `core/ports/training.py` — `TrainingProvider`, and the entities its methods speak in
2. `core/entities/training.py` — the dataset specification, the configuration, the status, the
   metric, the run's result
3. A **contract test suite** plus a fake provider that satisfies it, so M8-T03 inherits the tests
4. **ADR-0080** — the port's shape, and its relationship to the job runner
5. The import guard: nothing under `infrastructure/models/` may import training, and vice versa
   (ADR-0006's compliance clause, as a test rather than a review note)

**Out of scope**

- **`LocalTrainingProvider`** — M8-T03, and it brings ultralytics with it
- **The dataset builder** — M8-T02, which turns the project's annotations into what §1 specifies
- **Persistence of a run** — M8-T04
- **Any UI** — M8-T05
- **The remote protocol** — M8-T07. ADR-0006 already warns about designing it for an imagined
  deployment; the port is what the two share, and nothing more

---

## Definition of done

- [ ] `TrainingProvider` and its entities, with every method named by a caller M8 will actually write
- [ ] A contract suite a second implementation can be handed
- [ ] ADR-0080 + the ADR index
- [ ] `make check` green, golden byte-identical (nothing here computes a number)
- [ ] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [ ] Commit: `M8-T01: what a training run is, before anything trains`
