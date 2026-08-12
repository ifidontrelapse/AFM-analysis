# ADR-0042 — The index is in the database, the measurement table is a file

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T05)
- **Affects:** `infrastructure/storage`, `application/use_cases`, schema v2 · M4 · M4-T11's export

## Context

M4's second exit criterion is *"detection and measurement results round-trip through SQLite and
the filesystem"*. Until this task the pipeline returned a `PipelineResult` and the process exited
with it: nothing in the project could answer "what did we find in this scan last week?"

Two documents say different things about where the answer belongs. ADR-0003's storage rule lists
measurements among what SQLite holds; ADR-0003's *layout* says `results/` holds "detections,
measurements, generated masks" as files. And between them, ADR-0031 changed the thing being
stored: a measurement table is a **core plus blocks**, where a block is present in full or absent
in full and `method` names the producer — a shape chosen precisely so that a table is never one
wide grid with NULLs where a modality has nothing to say.

## Decision

### 1. The index is relational; the tabular product is a file

- **`analysis_runs`** and **`detections`** are tables (schema v2). Which analysis ran, on what,
  with which detector and mode, and every detection it produced. Fixed columns, and everything
  above queries them: the viewer draws detections, the exporter counts them.
- **The measurement table** is written to `results/run_<id>/measurements.csv`, and the run row
  carries its **relative** path like every other path in the project (ADR-0003).

The reason is ADR-0031. A relational schema for a table whose columns depend on the producer is
either one wide table with NULLs — which ADR-0031 rejected in those words, because it would say
SEM/TEM *has* heights and they are all missing — or an EAV pivot that nobody can read and that
loses the dtypes the schema declares. Neither is worth it for a table whose only query today is
"give me all of it".

This reading satisfies both halves of ADR-0003, and the exit criterion names the split itself:
*through SQLite **and** the filesystem*.

### 2. Nothing is written for a run that measured nothing

`detect` mode produces `empty_measurement_table(...)` — the right columns, no rows. No file is
written and `measurements_path` is `NULL`. An empty table on disk would be a measurement that
claims to have happened, and reading it back would report "no particles found" for a run that
never looked.

Reading a run whose file is *missing* raises `MissingFileError` rather than returning an empty
table, for the same reason.

### 3. Masks are not persisted

Segmentation needs SAM2 weights, which are not in this repository and not in the gate
(PROJECT_RULES §6). A storage format written now would be one that nothing under test can produce
— the definition of a format that turns out wrong.

The trigger to write it is named: the image viewer in M5, or annotations in M4-T07, whichever
first needs a mask to survive a session.

Consequences of not writing it, stated so nobody discovers them later: `results/` is **not**
covered by `check_integrity`, and a measurement file whose run row cascaded away stays on disk.
Derived data — the cost is a re-run, not a loss — and the day `results/` needs reconciling is the
day masks land in it.

### 4. Results cascade with their image

`REFERENCES images(id) ON DELETE CASCADE`, and the same from `detections` to `analysis_runs`.

ADR-0040 argued *against* deleting a row whose file went missing, and that argument does not
transfer: an image row is the expensive half of its pair, while a detection is **derived** — a
particle found in a scan the project no longer knows about is litter. This is also what finally
makes M4-T02's `PRAGMA foreign_keys = ON` load-bearing rather than a precaution.

### 5. One use case, and the loaders are imported directly

`run_analysis`, not `RunDetection` / `RunSegmentation` / `MeasureParticles`. The mode is a value
`PipelineConfig` already carries and `capabilities.py` already validates before anything runs;
three functions differing by a string literal is ADR-0041's case one task later.

It calls `run_preprocessing` and `load_microscopy_image` by name, as the two existing use-case
modules already do. The `ImageLoader` port is the right answer and it is not this task's:
`core/ports/__init__.py` dates it M2-T10 / M6, and introducing it here would mean rewriting two
call sites in the same commit as the first persistence code. **Debt, taken deliberately.** Its
trigger: the second loader implementation, or M6, whichever comes first.

### 6. The scale the project recorded is the scale the analysis uses

`run_analysis` passes the image row's `pixel_size_nm` into preprocessing, and `run_preprocessing`
gained the parameter to accept it.

This is not housekeeping. An `.npy` carries no metadata at all, so without it an image imported
*with* a known scale is analysed as though it had none: every `radius_nm` comes back `None` and
the physical minimum-size filter is silently skipped. That is the D-07 family of defect M3 spent a
milestone eliminating, reintroduced one layer up — and it was live until a test caught it. An
SPM's header still wins, because `load_afm` ignores the argument for that format.

## Consequences

**Positive**

- Results outlive the session, and the criterion is executed rather than asserted.
- The measurement table on disk is exactly what M4-T11 has to export, so the export becomes a
  copy-and-rename rather than a serialiser.
- The operator can open `results/run_000001/measurements.csv` in anything, which is ADR-0003's
  promise applied to derived data.
- Deleting an image cleans up after itself in the database.

**Negative**

- Two places to look for one run's output, and the file can go missing independently of its row.
  Mitigated by a loud error and by the fact that it is reproducible data.
- `results/` is outside the integrity check (§3), so nothing reports an orphaned measurement file.
- A run stores no parameters beyond detector and mode, so "what settings produced this?" is not
  answerable yet. Deliberate: `PipelineConfig` has twenty fields and freezing their storage before
  the model registry (M4-T13) exists would fix the wrong shape.

**Neutral**

- Schema version 2. The first migration applied to a database with rows in it — the case ADR-0039's
  mechanism was built for and had never run.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Measurements in one wide SQL table | ADR-0031 rejected exactly this shape for the in-memory table: NULLs would claim a modality has heights and they are missing |
| Measurements as EAV rows (`run_id, name, value`) | Loses the declared dtypes, and every read is a pivot, to serve a query nobody has asked for |
| Measurements as JSON in a column | A blob in a relational store, unreadable without our code — the opposite of ADR-0003's promise |
| Everything in files, no `analysis_runs` table | No way to ask "which images have been analysed?" without walking the directory, and the viewer asks that first |
| Three use cases, one per mode | Three names for `run_pipeline(mode=…)`; ADR-0041 settled the general case |
| Persist masks now | A format nothing in CI can produce, written blind against a model whose weights are not here |
| Keep results when their image is deleted | Rows referring to an image nobody can name, which no query can meaningfully return |

## Compliance

- `tests/integration/test_analysis_results.py` runs a characterization phantom through a real
  project, closes it, reopens it, and reads both halves back.
- The same file asserts a `detect` run writes no table, a deleted table raises rather than reading
  as empty, an absent `bbox` survives storage as `None`, and the cascade removes runs with their
  image.
- One test drives a **v1 database with rows in it** up to v2 and asserts the rows survived.
- The scale test fails if `run_analysis` stops passing the record's `pixel_size_nm` — the defect
  §6 describes is a red test, not a comment.
- No path stored by this task is absolute; the `CHECK` on `measurements_path` says so in SQL.

## References

- ADR-0003 (projects are directories) — both halves of it, reconciled here
- ADR-0031 (one measurement schema) — why the table is not relational
- ADR-0039 (the schema and its migrations) — the mechanism this task first exercised on real data
- ADR-0040 (the repository reports and does not reconcile) — why *this* deletion is different
- ADR-0041 (a use case earns its place) — the rule §5 applies
- ADR-0019 / ADR-0025 (an unknown scale is a state) — the invariant §6 restores
