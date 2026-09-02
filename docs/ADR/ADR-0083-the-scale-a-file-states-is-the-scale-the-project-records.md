# ADR-0083 — The scale a file states is the scale the project records

- **Status:** Accepted
- **Date:** 2026-09-02
- **Deciders:** operator (reported it), engineer
- **Affects:** `application/use_cases/projects.py`, `application/use_cases/preprocessing.py`,
  `gui/dialogs/import_options.py`, `gui/viewmodels/session.py` · M5-T07's import dialog,
  M7-T05's ruler

## Context

The import dialog asks two questions: the modality, and the pixel scale. The operator asked why
the second one is asked at all — *"during parsing of the `.00*` files it should be pulled out
automatically."*

It is pulled out. `_read_nanoscope_z` has read `Scan Size` and divided it by `Samps/line` since
M2-T04, and `load_afm(fmt="spm")` **ignores the `pixel_size_nm` it is handed** and returns the
header's. So the scale was being read on every load and thrown away at the one moment it could
have been recorded:

| Where | What it used | For a `scan.000` imported with the dialog left at *unknown* |
|---|---|---|
| The viewer, the properties panel | the loaded image's scale | **750 nm/px** — the header |
| `run_analysis`, `run_preprocessing` | the row's, which `load_afm` then ignores | **750 nm/px** — the header |
| The explorer's Scale column | the row's | *scale unknown* |
| `ruler_length`, `ruler_profile` | the row's | **no nanometres at all** |

Two of those four are wrong, and the fourth is the one an operator drags across a particle. The
row is the odd one out: everything that computes with a scale gets it from the file, and the row
was the only copy that could disagree — so it did.

`import_image` records what it is given, once, at import. Nothing has ever revisited it.

## Decision

**1. The import records the scale the file states.** `import_images` asks each AFM source for its
own scale (`stated_pixel_size_nm`) and records that. The dialog's answer is used for the files that
state nothing — an `.npy`, an SEM or TEM image — which is what the field was added for in M5-T07.

**Per file, not per batch:** a folder off the instrument can hold a 3 µm frame and a 500 nm one,
and one dialog cannot be right about both.

**The header wins over the answer.** Not a preference between two opinions: for an SPM the two are
not equal in standing, because `load_afm` reads the header and ignores its argument. Recording the
answer would put a number in the project that nothing computes with, and then show it in three
panels.

**2. Asking for a scale never refuses a file.** `stated_pixel_size_nm` returns `None` for anything
it cannot read — a truncated download, a missing file, a JPEG — and the import proceeds exactly as
it did. A batch that lost a file because a *scale lookup* had an opinion would be a new refusal
invented in a lookup (ADR-0041 keeps the batch policy).

**3. A measurement uses the scale of the array it is taken over.** `SessionViewModel` measures with
the loaded image's `pixel_size_nm`, not the row's. This is what fixes the projects imported *before*
this decision, without touching their database.

## Consequences

**Positive** — the four rows of the table above agree. A ruler over a Nanoscope scan reads
nanometres, in a project made today and in one made last month. The explorer's Scale column stops
saying *unknown* about files that state a scale. The dialog's remaining question is one the files
genuinely cannot answer.

**Negative** — the import reads every AFM file it copies, so importing forty scans now parses forty
headers (and, since `load_afm` has no header-only entry point, forty payloads). Measured on the
synthetic fixture the cost is microseconds per file and it is dominated by the `shutil.copy2`
beside it, but it is real work that did not happen before. A header-only read is the obvious fix
if a folder of 4096² scans ever makes it matter.

**Neutral** — rows written before this decision keep their `None`. Nothing backfills them: a write
triggered by opening a project is a surprise, and the measurement path no longer depends on it.
Backfilling is in the backlog as **B-073**.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Leave it: `load_afm` already ignores the row for an SPM | It is what the project did, and the ruler is the counter-example. Three surfaces read the row directly. |
| Ask the dialog only for formats that state no scale | The dialog is shown once for a batch, before any file is read; it would have to read them all to know what to ask, which is this decision plus a modal delay. |
| Backfill every row when a project opens | A write nobody asked for, on every open, over every file — and it would still leave the ruler reading a row instead of the array it is drawn on. |
| Record the operator's answer and let the header win only in the science | Two numbers, one shown and one computed with. This is the state that produced the report. |

## Compliance

- `tests/integration/test_project_lifecycle.py::TestTheScaleThatGetsRecorded` — the header wins,
  per file, and the answer is still used where a file states nothing.
- `tests/unit/test_afm_io.py` — `stated_pixel_size_nm` returns `None` rather than raising, for
  every unreadable shape a batch can contain.
- `tests/gui/test_ruler_tool.py::TestWhatIsStored` — a row written before this decision measures in
  nanometres anyway.

## References

- ADR-0025 — an absent scale is a state, never a fabricated value
- ADR-0026 — a header without a scan size parses
- ADR-0041 — one modality per import call, and a bad file costs only itself
- M5-T07 (the dialog), M7-T05 (the ruler), M2-T04 (the parser)
