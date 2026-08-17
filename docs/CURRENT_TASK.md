# CURRENT TASK

**ID:** `M7-T09`
**Title:** Annotation export/import in a training-ready format
**Milestone:** M7 — Annotation & metrology tools, ninth task
**Defect:** — · **ADR:** **ADR-0078**
**Branch:** `feat/m7-annotation-tools`
**Status:** **done 2026-08-17.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

It is **M7's fourth exit criterion** — *"annotations export to a format the M8 dataset builder
consumes"* — and the last thing standing between six milestones of hand work and the milestone that
trains on it. Everything M7 built writes rows into one project directory; nothing can take them out,
and nothing can bring labels in from the tools an operator already uses.

The reading that shapes the task: **an export of annotations is lossy and must say so.** A project row
carries an outline, a painted mask, a note, an id, two timestamps and — the one M8 depends on — the
`source` that says whether a person drew it or a detector did (ADR-0044). A YOLO label file carries a
class index and four numbers.

---

## The decisions this task has to make

**1. The format is YOLO's, because the trainer is ultralytics.**

`labels/<image stem>.txt`, one line per box, `class cx cy w h` normalised to the image and clamped to
`[0, 1]`, beside a `classes.txt` whose line numbers *are* the class indices. It is what M8-T03 will
train with, what labelImg, CVAT and Roboflow read and write, and therefore the one format that makes
*export* and *import* the same decision instead of two.

**2. `data.yaml` and the train/val split are M8-T02's, not this task's.**

A split is a *dataset* decision — how much to hold out, and stratified by what. Writing one here means
M8-T02 arrives to find its main decision already made by a task that had no reason to make it.

**3. Adopted boxes are a scope the menu names, not a default it hides.**

ADR-0044 built `source` because *"a model trained on its own output is confirming itself"*, and an
export that silently includes `from_detection` boxes is exactly how a training set stops being able to
tell. Two menu items, named — ADR-0067's rule, one milestone on: *hand-drawn only*, and *everything*.

**4. An import cannot guess where the labels came from, so it asks.**

A `.txt` file says nothing about who drew the box. M5-T07's dialog asks modality and pixel size for the
same reason and *"invents neither"*; this asks the one question a label file cannot answer, and writes
the file it came from into the annotation's `note`, so the provenance survives the trip.

**5. An import is one edit, on the main thread.**

Two hundred labels through `Composite` is one entry on the history and one `Ctrl+Z` (ADR-0077 §3) —
the first caller of that class outside adoption. It does **not** run as a job: the command stack is
deliberately not thread-safe, because undo is one person's sequence of actions (ADR-0045), and a job
that edits is the one shape ADR-0043 and ADR-0045 agree must not exist. Export *is* a job: it reads
every scan and writes files, and touches no history.

**6. What is refused, and what is only reported.**

A label file naming a class `classes.txt` does not have, or a coordinate outside `[0, 1]`, is a
**refusal**: it is not describing this image. A label file with no image of that name is **reported**
and skipped — ADR-0040's shape, because a directory holding labels for a bigger dataset is a normal
thing to import from, not a corrupt one.

---

## Scope

**In scope**

1. `application/use_cases/annotations.py` — `export_annotations`, `import_annotations`, and the two
   pure functions that format and parse a line
2. `core/ports/project_repository.py` + the SQLite adapter — `write_export_text`, since `write_export`
   is a `DataFrame` and a `.csv` suffix
3. `gui/viewmodels/session.py` — the export as a job, the import as one `Composite`
4. `gui/dialogs/` — the one question an import cannot guess
5. `gui/main_window.py` — two export items and an import item
6. **ADR-0078**
7. Tests: the line format exactly, normalisation and clamping, the source filter, a **round trip**,
   the refusals, the report, one undo for an import, and the menu wiring

**Out of scope**

- **`data.yaml`, the train/val split, copying images** — M8-T02's dataset, by decision 2
- **Exporting outlines or masks as segmentation labels** — a polygon's box is already stored beside
  its outline (ADR-0072), so the box export loses nothing the row did not already carry; segmentation
  labels wait for a trainer that asks for them
- **COCO/VOC** — a second format with no caller (ADR-0041's rule)

---

## Definition of done

- [x] Annotations export as YOLO labels + `classes.txt` under `exports/`
- [x] The same directory imports back, as annotations whose source the operator stated
- [x] Hand-drawn and everything are two named scopes — **in the menu and in the directory name**
- [x] ADR-0078 + the ADR index
- [x] `make check` green — 1305 tests, golden byte-identical, mypy unchanged at 6
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M7-T09: annotations out and back, in the format the trainer reads`

---

## What it turned up

**Two exports a second apart landed in one directory.** The test written for the source filter found
it: the hand-drawn label set and the everything set shared a timestamped name, so the file an operator
would then have trained on held both. The timestamp keeps ADR-0048's promise across time; it took the
**scope in the name** to keep it across the two menu items.

**The M6-T02 name guard caught the viewmodel saying *YOLO*.** Which trainer reads these files is
`application`'s business (PROJECT_RULES §2.5); the window offers *annotations*, in and out — and the
guard, written for a detection panel two milestones ago, is what said so.

**`write_export` could not be reused.** It takes a `DataFrame` and forces a `.csv`, and a label export
is a directory of small text files — so `write_export_text` sits beside it, sanitising **per path
component** because this one is allowed to name subdirectories.
