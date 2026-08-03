# ADR-0005 — Models are pluggable providers behind ports and a registry

- **Status:** Accepted
- **Date:** 2026-08-03
- **Affects:** `core/ports/`, `infrastructure/models/`, `application/capabilities.py` · M2, M4-T13

## Context

The project uses LoG (classical), YOLOv8 (learned detection) and SAM2 (segmentation)
today, and expects more models later — including models the operator trains themselves
(M8). Model choice is a scientific variable, not an implementation detail: comparing
detectors is part of the work.

The current design does not support that. `pipeline.py` selects with `if cfg.detector ==
"log" … elif "yolo" … else raise`. Weights are a hardcoded string in a config dataclass
(`yolo_model_path = "./checkpoints/best12x.pt"`). The permitted combinations of modality
× detector × mode exist in three places — `pipeline.py`, `ConfigPanel.tsx` and a prose
table — that already disagree with each other (audit §4). Adding a model today means
editing the dispatcher, the config dataclass, the UI, and the documentation.

## Decision

Models are **providers behind ports**, resolved through a **registry**.

1. `core/ports/` defines `Detector` and `Segmenter` as protocols. They speak the domain's
   language — arrays, `PixelScale`, `Detection`, masks — and know nothing about torch.
2. Each implementation lives in `infrastructure/models/<name>/` and is responsible for
   everything model-specific: input preparation, weight loading, inference, output
   conversion.
3. A **registry** maps an identifier to a provider factory. Selection is a lookup, never
   an `if/elif`.
4. Weights are described by a **`ModelDescriptor`** persisted in the database: id, task,
   framework, path, input size, class map, provenance, checksum, training run. Not a raw
   path string in a config object.
5. The **capability matrix** (modality × detector × mode) has exactly one owner:
   `application/capabilities.py`. The GUI disables options *because the matrix says so*.
   Validation runs **before** any inference (fixes D-14).

**No model-specific logic outside its provider.** The strings `"yolo"`, `"sam2"`, `"log"`
must not appear in `gui/` or in `core/science/`.

## Consequences

**Positive**

- Adding a model is: implement the port, register one line, add a capability row.
- Model-specific defects stay contained. The YOLO input-preparation defect (D-03,
  12.6% of dynamic range retained) is currently indistinguishable from a detection-quality
  problem; behind a provider it is one file with one contract test.
- Operator-trained models (M8) become first-class without a code change.
- The UI and the backend can no longer disagree about what is allowed, because there is
  only one rule.
- A plugin system later (B-002) becomes registration from an entry point, not a redesign.

**Negative**

- The port must be general enough for models we have not seen. `sizes=` is LoG-specific
  today and leaks into the call site — designing that away costs a `DetectionContext`
  parameter object that is slightly over-general for two implementations.
- One more layer between "run YOLO" and running YOLO.
- Registry lookups fail at runtime, not at type-check time, unless the identifiers are
  constrained.

**Neutral**

- `PipelineConfig` loses its model-specific fields; they move into per-provider settings
  addressed by descriptor.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Keep the `if/elif` dispatcher | Every new model touches the dispatcher, the config dataclass, the UI and the docs. Already unmanageable at two detectors. |
| Subclass hierarchy (`BaseDetector` inheritance) | Half-exists today and works, but couples implementations to a shared base and encourages putting model logic in the parent. Protocols + composition keep providers independent (PROJECT_RULES §2.9). |
| Configuration-file-driven model loading (no registry) | Moves the dispatch into YAML without removing it; no type safety, and errors surface as missing keys. |
| Full plugin system now | Solves a problem we do not have yet (B-002). The registry is the 20% that makes the other 80% additive later. |

## Compliance

- Grep gate: `"yolo"`, `"sam2"`, `"log"` as behaviour selectors appear only in
  `infrastructure/models/` and `application/capabilities.py`.
- Adding the next model must not modify any file outside `infrastructure/models/<name>/`,
  the registry line, and the capability table — enforced at review.
- Contract tests run the same suite against every registered provider.
- Capability validation is asserted to run before the detector is constructed.

## References

- `systempromt.md` (Models)
- `docs/Architecture.md` §4.1
- `docs/audit/2026-07-28-baseline-audit.md` D-14, §4 "The capability matrix has no owner"
- `docs/TASKS.md` M2-T08, M2-T10, M4-T13
