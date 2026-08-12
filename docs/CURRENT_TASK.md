# CURRENT TASK

**ID:** `M4-T13`
**Title:** A model is a record, not a path in a default argument
**Milestone:** M4 — Application layer, thirteenth task
**Defect:** W10 (model paths are hardcoded config strings) · **ADR:** ADR-0005 is accepted;
**ADR-0050** records what implementing it decided
**Branch:** `feat/m4-application-layer`
**Status:** **done 2026-08-12.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

W10, still open and still visible in two default arguments:

```python
yolo_model_path: str = "./checkpoints/best12x.pt"     # PipelineConfig
def __init__(self, model_path: str = "./checkpoints/best12x.pt", ...)   # YoloDetector
```

A relative path to a file nobody promises exists, with no version, no checksum, no record of what
it was trained on, repeated in two places. ADR-0005 decided the replacement — a `ModelDescriptor`
in the database and a registry keyed by a string — and the milestone's fourth exit criterion is
*"model registry resolves `yolo` and `sam2` to providers via `ModelDescriptor`"*.

M4-T12 also left a gap by name: the resolved device reaches nothing yet. The registry is where a
provider gets constructed, so it is where a device can be handed over.

---

## The decisions this task has to make

**1. What is registered — a class, or a factory?** A **factory**, and the registry never
constructs anything by itself.

Building a `YoloDetector` loads weights off disk; a registry that instantiates on lookup makes
"what models do I have?" an expensive question and an impossible one in CI, where no weights
exist. `resolve()` returns something callable, and the caller decides when to pay.

**2. Where do weights live, and may a path be absolute?** Both, and the *reason* is the decision.

Project-local weights go in `models/` and are stored relative, like every other path (ADR-0003).
But nobody copies a 137 MB checkpoint into every project, so an absolute path to a shared file is
**allowed** — and a project carrying one is honest about the consequence: it opens on another
machine, and that model is simply unavailable there. Refusing absolute paths would mean either
duplicating gigabytes or lying about where the file is.

**3. What identifies a model?** A caller-chosen `model_id` string, unique within the project — the
same string ADR-0005 puts in configuration. Not a hash: an operator names their model, and a
checksum answers a different question ("is this the file I recorded?"), which the descriptor also
carries.

**4. Does this task rewire `run_pipeline`?** **No.** Its `if/elif` on `cfg.detector` is what the
`Detector` port removes, and doing it here would mean a behaviour-preserving refactor of the one
function the golden covers most, bundled into a commit about storage — exactly what ADR-0010
forbids. The registry is additive; the swap belongs with the composition root that will call it
(M5).

**5. Do the defaults change?** No. `PipelineConfig.yolo_model_path` keeps its value, because the
golden records that field and M4 must not move a number. What changes is that there is now a
**better** way to say which model, and W10 closes when the GUI uses it rather than when the
default disappears.

---

## Scope

**In scope**

1. `ModelDescriptor` in `core/entities/model.py` — id, task, framework, path, input size, class
   map, provenance, checksum
2. Migration step 5: the `models` table, and repository methods to register / list / get
3. `infrastructure/models/registry.py` — framework → factory, `resolve(descriptor, device)`
4. **ADR-0050** — factories not instances, absolute paths allowed with their consequence, the
   `run_pipeline` rewiring deferred with a trigger
5. Tests: the descriptor round-trips, an unknown framework is refused by name, `yolo` and `sam2`
   resolve to their providers **without loading weights**, and a device is passed through

**Out of scope**

- **Rewiring `run_pipeline`** — decision 4
- **Downloading or validating weights.** Verifying a checksum is a read of a large file; the
  descriptor stores it so somebody *can*, and who asks is a question for M5
- **Training** — M7/M8 produce descriptors; this task stores them

---

## Definition of done

- [x] `ModelDescriptor`, schema v5, repository methods, the port extended
- [x] A registry resolving `yolo` and `sam2` to providers without constructing them
- [x] ADR-0050
- [x] Tests, including a device handed to a factory
- [x] `make check` green — golden byte-identical
- [x] Docs, the ADR index, the roadmap criterion
- [x] Commit: `M4-T13: a model is a record, not a path in a default argument`

---

## What it turned up

**The guard written last session fired on the very next migration.** `TABLES_BY_VERSION` in
`tests/integration/conftest.py` did not list v5's `models` table, and its own self-check went red
immediately — instead of three unrelated migration tests failing later with
`CREATE TABLE … already exists`. One task after it was written, it paid for itself.

**W10 is made closable rather than closed, and saying which is the point.** The registry exists and
works; `PipelineConfig.yolo_model_path` still holds a path, because deleting it means rewiring the
function the golden covers most. Half a defect closed, with the other half assigned, beats a
commit that quietly does both.
