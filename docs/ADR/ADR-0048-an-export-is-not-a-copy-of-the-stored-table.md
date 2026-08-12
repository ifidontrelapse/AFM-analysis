# ADR-0048 — An export is not a copy of the stored table

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T11)
- **Affects:** `application/use_cases/export`, `infrastructure/storage` · M4 · M5's export action

## Context

ADR-0042 predicted this task would be nearly free: the measurement table is already a CSV, at
`results/run_<id>/measurements.csv`, so exporting it could be `shutil.copy`.

Writing it showed the prediction was half right. The *format* is free. The export is not, because
storage and export answer different questions.

## Decision

**An export is a document for a person; the stored table is an index for the application.** Three
differences follow, and each of them is a line of code that a copy would not have.

### 1. Provenance columns, in front

`image`, `image_id`, `run_id`, `detector`, `mode`, `pixel_size_nm` are prepended to every row.

The stored table does not repeat any of it, because it is *filed under* the run that produced it —
the directory is the context. A CSV on somebody's desktop has no context at all, and a column of
heights with no scan name is a column of numbers. In front rather than appended, because a
spreadsheet opens on column A and the first question is always which scan this is.

`pixel_size_nm` is empty when the scale is unknown, never `0`: a spreadsheet reading a zero there
would compute nanometres from a pixel count (ADR-0025, one layer out).

### 2. More than one run in one file

`export_measurements(repo)` with no argument exports every run of every image, concatenated.

Statistics across a dataset is the reason the measurements exist. Assembling it by hand from twelve
files is not a workflow, and it is the step where an operator drops one.

### 3. A name that does not overwrite yesterday's

`measurements_<what>_<timestamp>.csv` by default. An export is a snapshot, two in one day are the
normal case, and silently replacing the first loses work the operator believes they have.

A caller-supplied name is reduced to something a filesystem accepts, in the **repository**: the
name arrives from an operator's text field, and a `/` in it would write outside `exports/`. The
sanitisation is where the write happens, not where the name is chosen, because that is the funnel
every future export passes through.

### 4. Nothing dishonest gets written

- If no selected run produced a measurement table — a detect-only run measures nothing — the export
  **raises** instead of writing headers with no rows. An empty CSV is indistinguishable from *"we
  measured and found nothing"*, which is a different statement about the sample.
- If a run's stored table is missing, the failure propagates (ADR-0042 already made that call). An
  export silently short by one scan of twelve is a wrong dataset that looks right.

### 5. The split of labour

The use case decides what an export *contains*; `write_export` on the repository decides where it
lands and what it may be called. `application` may not touch the filesystem (Architecture §3.2),
and this is the smallest seam that respects it — one port method, no new abstraction.

## Consequences

**Positive**

- The file an operator opens explains itself without the application next to it.
- One command produces the whole project's measurements, which is the shape the science needs.
- Exporting cannot quietly produce a file that misrepresents what happened.
- CSV stays the only export format, as PROJECT_RULES §5 requires.

**Negative**

- Concatenating tables from different producers gives a union of columns with blanks where a
  producer has none — a baseline run and a SAM2 run in one file. That is ADR-0031's shape crossing a
  file boundary; the `method` column says which row is which, and splitting by producer is a
  decision for whoever asks for it.
- Everything is held in memory during the concatenation. Fine for the sizes in front of us; a
  project with a million rows would want streaming, and that is a change to one function.

**Neutral**

- Nothing is recorded about an export in the database. A file in `exports/` is the record, and
  `exports/` is outside the integrity check for the same reason `results/` is (ADR-0042 §3).

## Alternatives considered

| Alternative | Why not |
|---|---|
| Copy the stored CSV | No provenance, one run at a time, and a name that says nothing |
| Store the provenance columns in the file `results/` holds | Repeats in every row what the directory already says, and bloats the index for the export's benefit |
| Write an Excel workbook | A dependency, and PROJECT_RULES §5 says CSV only, for now |
| One file per run in a zip | The operator then merges twelve files by hand — the step §2 exists to remove |
| Export empty when nothing was measured | A file that says "no particles" about a run that never looked |
| Sanitise the name in the use case | The check belongs where the write is, or the next caller of `write_export` has to remember it |

## Compliance

- `tests/integration/test_export.py` asserts the provenance columns and their order, two runs in
  one file, the measured columns surviving, a relative path under `exports/`, a name that cannot
  escape the directory, two exports not colliding, the refusal on a detect-only run, and the loud
  failure on a missing table.
- No exported path is absolute, and no export is written outside `exports/`.

## References

- ADR-0042 (the index is in the database, the measurement table is a file) — the prediction this
  task tested
- ADR-0031 (one measurement schema) — why a concatenation has a `method` column
- ADR-0025 (an unknown scale is a state) — why `pixel_size_nm` is empty rather than `0`
- `PROJECT_RULES.md` §5 — CSV only, for now
