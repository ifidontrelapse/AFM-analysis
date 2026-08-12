# ADR-0050 — A model is a record, and the registry hands back factories

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T13)
- **Affects:** `core/entities`, `core/ports`, `infrastructure/models`, schema v5 · M5 · M7/M8

## Context

W10, open since the audit and visible in two default arguments:

```python
yolo_model_path: str = "./checkpoints/best12x.pt"          # PipelineConfig
def __init__(self, model_path: str = "./checkpoints/best12x.pt", …)   # YoloDetector
```

A relative path to a file nobody promises exists, with no version, no checksum and no record of
what it was trained on — written twice. ADR-0005 decided the replacement and this task builds it,
against the milestone's fourth exit criterion: *"model registry resolves `yolo` and `sam2` to
providers via `ModelDescriptor`"*.

M4-T12 also left a gap by name: it resolves a device that nothing consumes.

## Decision

### 1. The registry hands back factories, never instances

`resolve(descriptor)` returns a callable. Nothing is read from disk until somebody calls it.

Constructing a detector loads weights, so a registry that instantiates on lookup makes *"what
models does this project have?"* an expensive question — and an impossible one in CI, where no
weights exist at all. The exit criterion is met in a test that resolves a model whose file is not
there.

### 2. Weights may be inside the project or shared, and the consequence is stated

A path under the project's `models/` is stored **relative**, like every other path (ADR-0003). An
absolute path to a shared checkpoint is stored as it is, and `ModelDescriptor.is_external` says so.

Nobody copies a 137 MB checkpoint into every project. Refusing absolute paths would force either
duplicating gigabytes or lying about where the file is; allowing them costs one honest sentence:
**a project referencing an external model opens on another machine, and that model is unavailable
there.** The `models` table is the one place in this schema without the `NOT LIKE '/%'` check, and
its comment says why.

### 3. A model is identified by a name somebody chose

`model_id`, unique in the project, and the same string a configuration names. Not a hash: an
operator names their model, and the checksum answers a different question — *is this still the file
I recorded?* — which the descriptor also carries, unverified, because checking it is a read of a
very large file and whose job that is belongs to whoever asks.

Registering an id twice **replaces**: retraining produces a new file under the name the
configuration already uses, and two rows for one id would make "which one" a question nobody can
answer.

### 4. `provenance` is free text

"trained 2026-08-01 on 412 annotations", "downloaded from …". A schema for provenance is a schema
somebody cannot fit their case into, and **provenance that must fit a schema stops being
recorded**. M7 will write it from a training run; a person will write it by hand; both are the
point.

### 5. The device reaches the provider here

The factory takes the resolved `Device` and passes `device.torch_name` to the provider —
`YoloDetector` gained a `device` parameter defaulting to `None`, which is ultralytics' own default,
so a caller that does not choose is unaffected.

This closes the gap ADR-0049 named. No provider asks torch where it should run; the manager
decides, the registry carries, the provider accepts (ADR-0004, PROJECT_RULES §2.6).

### 6. `run_pipeline` is **not** rewired, and the defaults do not change

Its `if/elif` on `cfg.detector` is what the `Detector` port removes, and doing it here would mean a
behaviour-preserving refactor of the function the golden covers most, bundled into a commit about
storage — exactly what ADR-0010 forbids. `PipelineConfig.yolo_model_path` keeps its value, because
the golden records that field.

So **W10 is not closed by this task; it is made closable.** The registry is additive, and the
swap belongs with the composition root that will call it (M5). Written down so the remaining half
is a plan rather than an oversight.

## Consequences

**Positive**

- A model has a version, a checksum and a provenance, and a project records which models it can
  use — the four things a path in a default argument does not have.
- Adding a framework is a provider plus one `register(...)` line, which is what ADR-0005 promised.
- The exit criterion is met without weights, in CI, because factories are cheap.
- M4-T12's device finally reaches inference.

**Negative**

- Two ways to say which model — the descriptor and the old default — coexist until M5 uses the
  first. That is the cost of not bundling a refactor into this commit, and §6 names who pays it.
- `ModelFactory` returns `Any`: a detector and a segmenter have different shapes, and unifying them
  would be an interface invented for symmetry. `ModelDescriptor.task` is what says which a caller
  asked for.
- Nothing verifies the checksum, so a swapped weight file is silent until somebody looks.

**Neutral**

- Schema version 5. The SAM2 factory returns a predictor rather than a wrapper class, because the
  SAM2 code here is functions taking a predictor — a wrapper would be an abstraction with one
  caller and no second implementation.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Registry returns constructed providers | Listing a project's models loads every weight file; impossible in CI |
| Registry keyed by a free string | The descriptor already carries a typed framework; two vocabularies drift |
| Weights must live inside the project | Duplicating gigabytes per project, or lying about where the file is |
| Identify a model by its checksum | An operator names their model; a checksum answers "did it change", which is a different question |
| A structured provenance schema | Provenance that must fit a schema stops being recorded |
| Rewire `run_pipeline` in this commit | A golden-covered refactor bundled into a storage change — ADR-0010's rule |
| Delete `yolo_model_path` now | The golden records the field; M4 does not move a number |

## Compliance

- `tests/integration/test_model_registry.py` covers the round trip, replacement by id, an unknown
  id named in the message, relative and external paths, survival across a session, both frameworks
  resolving, resolution of a model whose file does not exist, an unknown framework refused with the
  list of what is known, and **the device arriving at the provider**.
- No module outside `infrastructure/models` names ultralytics or SAM2.
- `models` is the only table permitted an absolute path, and its comment says why.

## References

- ADR-0005 (models are pluggable providers behind ports and a registry) — the decision implemented
- ADR-0049 / ADR-0004 — the device this closes the loop on
- ADR-0010 (one defect, one commit) — why `run_pipeline` is untouched here
- `docs/Architecture.md` §2.3 W10, §4.1 · `docs/Roadmap.md` M4 exit criteria
