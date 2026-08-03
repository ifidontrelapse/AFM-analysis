# ADR-0001 — Clean Architecture with a preserved scientific domain

- **Status:** Accepted
- **Date:** 2026-08-03
- **Affects:** whole codebase · M2

## Context

The repository is a 2 021-line research library plus notebooks. It has no application
layer, no persistence, no UI, and no seam where one could be added: `pipeline.py`
dispatches detectors with `if/elif`, modality is inferred from `isinstance`, and
`src/__init__.py` creates five import cycles so that importing the "dependency root"
loads 1179 modules (audit D-18).

At the same time the science is good and measured: the flatten → substrate → detect →
measure chain finds 24/24, 30/30 and 14/14 particles on the clean phantoms, and its
current behaviour is pinned by a golden file to `rtol=1e-6`.

We need an architecture that supports years of feature work — projects, annotation,
training, model management — without the UI, the storage format, or the model runtime
leaking into the numerics.

## Decision

Adopt Clean Architecture with four rings and one composition root:

```
gui  →  application  →  core  ←  infrastructure
                ↑
               app  (wires everything)
```

- **`core`** — entities, value objects, ports, and the scientific algorithms. Pure
  Python + NumPy/SciPy/scikit-image/pandas. No Qt, no torch, no SQLite, no filesystem,
  no network, no `print`.
- **`application`** — use cases, the capability matrix, jobs, the undo/redo command
  stack, DTOs. Depends on `core` only.
- **`infrastructure`** — adapters implementing `core` ports: SQLite, filesystem, model
  runtimes, device management, logging.
- **`gui`** — PySide6 views and viewmodels. Talks to use cases. Contains no business logic.
- **`app`** — the only place that constructs concrete implementations.

The existing scientific pipeline becomes `core/science`, **moved rather than rewritten**.
Model-backed code (YOLO, SAM2) is not domain — it imports torch — and moves to
`infrastructure/models`.

## Consequences

**Positive**

- The science becomes testable without a UI, a database, or a GPU — which is what makes
  the M3 defect work tractable at all.
- Storage format, model runtime and UI toolkit each become replaceable in one place.
- The dependency rule is mechanically checkable, so it does not decay into a code-review
  opinion.
- Training, plugins, batch processing and a future non-Qt frontend all become additive.

**Negative**

- More indirection than a research library needs today: a call that was one function
  becomes port → use case → provider. Real cost, paid on every feature.
- Ports must be designed before their second implementation exists, which risks
  designing for a use case that never arrives.
- The move itself is a large, low-glamour refactor (M2, 16 tasks) that produces no
  visible feature.

**Neutral**

- Import paths change everywhere; notebooks must be updated or retired.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Keep the flat library, add a UI on top | The UI would import the science directly, and every model, device and storage decision would end up in widgets. This is the failure mode the project is explicitly trying to avoid. |
| MVC / MVVM only, no domain ring | Sufficient for the UI, silent about model plugability, device selection, and the storage boundary — three of the five hard requirements. |
| Hexagonal architecture | Substantively the same ports-and-adapters idea. Clean Architecture is named in the project brief and its layer vocabulary is more widely understood. |
| Rewrite the science while restructuring | Would destroy the one asset the project has, and make every numerical delta unattributable. Rejected outright (ADR-0010). |

## Compliance

- An import-graph test (M2-T09) fails the build if `core` imports `gui`, `application`
  or `infrastructure`, or if it imports Qt or torch.
- `import nanoscope.core.entities` must load fewer than 100 modules.
- A lint rule (M5-T11) forbids `core.science` and `infrastructure` imports inside `gui/`.
- `python tests/characterization/capture.py` must report zero drift after every move.

## References

- `docs/Architecture.md` §3
- `docs/audit/2026-07-28-baseline-audit.md` D-14, D-18
- `docs/TASKS.md` M2
