# CURRENT TASK

**ID:** `M4-T05`
**Title:** Analysis results that survive the session — and the first real migration
**Milestone:** M4 — Application layer, fifth task
**Defect:** — (W1: no application layer exists) · **ADR:** **ADR-0042** (to be written)
**Branch:** `feat/m4-application-layer` — M4 changes no scientific output (PROJECT_RULES §7)
**Status:** planned 2026-08-12, implementation next.

---

## Why this task is next

M4-T04 met the milestone's first exit criterion. This one owns the second:

> *Detection and measurement results round-trip through SQLite and the filesystem.*

Nothing in the project can currently answer "what did we find in this scan last week?" — the
pipeline returns a `PipelineResult` and the process exits with it. This is also **the first task in
M4 that calls the scientific core**, so it is the first where a golden difference is even possible;
if one appears, the bug is in M4.

---

## The decisions this task has to make

**1. Three use cases, or one?** One: `run_analysis`.

`RunDetection`, `RunSegmentation` and `MeasureParticles` are `run_pipeline` with `mode` set to
`"detect"`, `"segment"` and `"baseline"` — the mode already lives in `PipelineConfig`, and
`capabilities.py` already validates the combination before anything runs. Three functions differing
by a string literal is ADR-0041's case again, one task later: a second name for the same method.

**2. Where do results live?** Split, and the exit criterion names the split itself — *"through
SQLite **and** the filesystem"*:

| | Where | Why |
|---|---|---|
| which analyses ran, and what they found | SQLite: `analysis_runs`, `detections` | Fixed columns, queried by everything above — the viewer draws detections, the exporter counts them |
| the measurement **table** | `results/<run>/measurements.csv` | ADR-0031 made it *variable by construction*: a core plus blocks that are present in full or absent in full, with `method` naming the producer. A relational table for that is either wide with NULLs — which ADR-0031 rejected in these words — or an EAV pivot nobody can read |

ADR-0003's layout says `results/` holds "detections, measurements, generated masks" as files, while
its storage rule says SQLite holds measurements. The reading that satisfies both, and this task's
decision: **the index is in the database, the tabular product is a file.**

**3. What about masks?** Not persisted. Segmentation needs SAM2 weights, which are not in this
repository and not in CI (PROJECT_RULES §6), so a mask format written now would be a format nothing
can produce under test. The trigger to write it is named in the ADR: the viewer in M5, or
annotations in M4-T07.

**4. What happens to an image's results when the image row is deleted?** They go with it —
`REFERENCES … ON DELETE CASCADE`. A detection of a particle in a scan the project no longer knows
about is not data, it is litter, and it is the *derived* half of the pair, so ADR-0040's argument
for keeping the row does not apply here. This is also what finally makes M4-T02's
`PRAGMA foreign_keys = ON` load-bearing rather than a precaution.

**5. How does the use case get the pixels?** Through the loaders in `infrastructure.storage`,
imported directly, exactly as `use_cases/preprocessing.py` and `use_cases/pipeline.py` already do.

An `ImageLoader` port is the correct answer and it is **not** this task's: the ports table dates it
M2-T10 / M6, and introducing it here means rewriting the two existing call sites in the same commit
as the first persistence code. Debt, taken deliberately, recorded in the ADR with its trigger
rather than discovered later.

**6. Which schema version?** **2**, through ADR-0039's mechanism — the first migration applied to a
database that already has rows in it. That is the property the mechanism was built for, and it has
never been exercised.

---

## Scope

**In scope**

1. Migration step 2: `analysis_runs` and `detections`, with foreign keys and cascade
2. `AnalysisRun` in `core/entities/project.py`
3. Repository: `save_analysis`, `runs_for`, `detections_for`, `measurements_for`, and the port
   extended to match
4. `application/use_cases/analysis.py` — `run_analysis(repository, image_id, config, predictor)`
5. **ADR-0042** — one use case not three, the index/product split, no masks yet, the cascade
6. Tests: the migration over a populated v1 database, the round trip, the cascade, and an
   end-to-end run over a characterization phantom written into a real project

**Out of scope**

- **Masks** — decision 3
- **CSV export** — M4-T11. Writing `measurements.csv` here is storage, not export: the export
  service decides what an operator's file looks like and where it goes
- **Jobs and progress** — M4-T06. `run_analysis` is synchronous
- **The `ImageLoader` port** — decision 5
- **YOLO and SAM2 paths under test.** Neither is in the gate (PROJECT_RULES §6); the LoG detector
  is the one that runs in CI, and the persistence code is detector-agnostic

---

## Expected blast radius

- **Zero golden differences**, and this is the first task where that claim is not free — the
  science is called. `run_pipeline` is not modified: the use case calls it and stores what comes
  back
- One migration step, two tables, one new application module, one ADR
- No new dependency

---

## Definition of done

- [ ] Schema v2, reached by a migration over a database with rows in it
- [ ] `save_analysis` / `runs_for` / `detections_for` / `measurements_for`, on the port too
- [ ] `run_analysis`, taking mode from the config rather than existing three times
- [ ] ADR-0042
- [ ] Tests: migration, round trip, cascade, end-to-end over a phantom
- [ ] `make check` green, golden byte-identical
- [ ] `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, `Roadmap.md`, ADR index
- [ ] Commit: `M4-T05: what the analysis found outlives the session`
