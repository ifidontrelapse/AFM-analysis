# CURRENT TASK

**ID:** `M2-T02`
**Title:** Extract entities and value objects
**Milestone:** M2 — Domain extraction (behaviour-preserving)
**Status:** **done 2026-08-04**. Rewritten for `M2-T03` at the start of the next session.
**Branch to use:** `feat/core-entities`
**Estimated size:** M
**Risk to scientific output:** **the first task in the project that moves scientific code.**
The golden is the gate, and it records `sorted(field names)` of `PipelineConfig` and
`PipelineResult` — so adding a field is drift, not just a change
**Selected:** 2026-08-04

---

## Why this task is next

`nanoscope/` exists but is empty (M2-T01). `src/types.py` is the dependency root of the
whole `src/` package — every other module imports from it and it imports from none of them
— which makes it the only module that can move without dragging another with it.

---

## Scope

Two commits, deliberately separate. If the golden goes red, which commit did it must be
obvious without bisecting.

> **It became three.** The strict `nanoscope.*` override rejects legacy code arriving
> verbatim, which this plan did not anticipate; the fix is its own commit rather than
> smuggled into the move. See `Progress.md`.

**Commit 1 — the move. Behaviour-preserving, no new names.**

1. The six dataclasses in `src/types.py` → `nanoscope/core/entities/`:
   - `image.py` — `AFMRawData`, `MicroscopyData`, `PreprocessingResult`
   - `detection.py` — `Detection`
   - `pipeline.py` — `PipelineConfig`, `PipelineResult`
   - `__init__.py` re-exports all six; that is the layer's public surface
2. `src/types.py` becomes a **re-export shim** — no class definitions. There must be
   exactly one `Detection` class object in the process, or `isinstance` starts lying and
   the five `src/` importers silently split into two type systems
3. Class bodies are copied character for character. No field added, renamed or reordered;
   no annotation "improved". The golden compares field *names*, and `dataclasses.asdict`
   ordering is field order

**Commit 2 — the new value objects. Additive, wired to nothing.**

4. `nanoscope/core/values/`: `Modality`, `Polarity`, `PixelScale`, `DeviceKind`
5. **Defined, not adopted.** Nothing in `src/` starts using them in this task. Replacing
   `modality: str` with `modality: Modality` changes what `dataclasses.asdict` produces
   and would move the golden — that adoption belongs to M2-T10 (capability matrix),
   M3-T10 (polarity) and M4-T12 (device), each of which has a consumer for it

**Out of scope**

- Adopting the new value objects anywhere — see above. They are deliberately unused, so
  **M2-T13 must not mistake them for dead code**; this file and `Progress.md` are the record
- Changing the pandas dependency in `PipelineResult.measurements`. `core` importing pandas
  is worth revisiting, but it is today's design and moving it is M2-T09's import-weight work
- Any other `src/` module. `preprocess.py` is M2-T03, `afm_io.py` is M2-T04

---

## Definition of done

- [x] `nanoscope/core/entities/` holds the six dataclasses; `nanoscope/core/values/` the
      four new value objects
- [x] `src/types.py` defines no class — only re-exports
- [x] `src.types.Detection is nanoscope.core.entities.Detection` — one class, not two
- [x] The five `src/` importers and `tests/characterization/capture.py` are untouched
- [x] `make check` green; **golden zero drift** — this is the whole point of the milestone
- [x] mypy: `nanoscope` **0 errors** under the strict override; `src` 21, not the 22
      predicted — one of them was `src/types.py:63`, which left with the move and is now
      the scoped D-16 ignore in `nanoscope/core/entities/detection.py`
- [x] CI green — run 16
- [x] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [x] Commits: `M2-T02: move the shared dataclasses into nanoscope.core.entities` and
      `M2-T02: add the Modality, Polarity, PixelScale and DeviceKind value objects`

---

## Plan

1. Branch `feat/core-entities`
2. Copy the classes into `core/entities/`, replace `src/types.py` with the shim
3. `make check` — **golden must be zero drift.** Run it before writing any new code, so the
   move is proven alone
4. Commit 1
5. Add the value objects; `make check` again; commit 2
6. Push, confirm CI, merge

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Two `Detection` classes exist** — one imported via `src.types`, one via `nanoscope`, and `isinstance` fails between them | The shim re-exports; it defines nothing. Asserted directly in the DoD. |
| A field is "tidied" during the copy | The golden records `sorted(f.name for f in fields(...))` for `PipelineConfig` and `PipelineResult`, so this is caught — but caught late. Copy the bodies verbatim and diff them before running anything. |
| The new value objects arrive unused and M2-T13 deletes them as dead code | Recorded here and in `Progress.md`, with the task that adopts each one named. |
| Import cycles get worse: `src/__init__.py` → `pipeline` → `src.types` → `nanoscope` | The shim adds one edge out of `src/` and none back in, so the existing five cycles (D-18) are unchanged. M2-T09 measures them. |

---

## Notes for the next session

`M2-T03` — move `preprocess.py` into `core/science/preprocessing/`. Unlike this task it
moves *behaviour*, not just declarations, so the golden stops being a formality.

Carried, still not tasks: **B-058** (the golden compares CPython exception text — ADR needed
before any Python upgrade), **B-054** (two README figures over 1 MB, M9-T01).
