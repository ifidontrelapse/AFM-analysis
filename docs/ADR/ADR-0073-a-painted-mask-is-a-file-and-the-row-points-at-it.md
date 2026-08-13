# ADR-0073 — A painted mask is a file, and the row points at it

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** operator + agent (M7-T04)
- **Affects:** `core/entities`, `infrastructure/storage`, `gui` · M7 · M8

## Context

M7-T03 gave an annotation an outline. The brush is the third shape and the first that is not a
handful of numbers: a painted mask is an array the size of the scan.

PROJECT_RULES §5 decided where it goes before this task existed — *"no mask bitmaps in the database;
masks are files, the database stores paths"* — and `docs/ProjectFormat.md` has had `annotations/`
set aside for them since M4-T01. So the decisions here are about **storage and honesty**, not about
the brush.

## Decision

### 1. The mask is a file; the row keeps its path

`annotations/mask_<id>.png`, relative to the project root like every other stored path (ADR-0003),
written **after** the row exists because the id is the name — `save_analysis`'s own sequence
(M4-T05). Schema **v7** adds one nullable column.

PNG rather than `.npy`: a mask an operator painted is a picture of their judgement, and a format
every image viewer on their machine can open is worth more than a few bytes. Written as 0/255, read
back as a boolean, so nothing downstream has to remember the convention.

### 2. The box is derived from the painted pixels

Exactly as ADR-0072 derives it from an outline. The three shapes now agree on one rule: **whatever a
reader wants as a box, the repository computes from what was actually drawn.** The far edge is
inclusive by one pixel, so a single painted pixel is a box with area rather than the zero-area one
the `CHECK` refuses.

### 3. Undoing an add removes the row and leaves the file

ADR-0040's rule, third application: forgetting a thing and deleting a file are different decisions.
A redo points the restored row at the same file, and a file with no row is exactly what
`check_integrity` reports.

### 4. Nothing is painted into the scan

The brush paints into a mask of its own; the pixmap the viewer shows is never touched. The viewer is
showing the file (ADR-0056), and a tool that edited the data an operator is measuring would be the
worst version of this feature.

### 5. A stroke that painted nothing stores nothing

Quietly, like an accidental click (ADR-0071 §4). The repository refuses an empty mask outright,
because a mask with no pixels has no shape.

### 6. A missing mask file is a refusal, not an empty mask

An empty one would read as *"the operator painted nothing"*, which is a different statement — the
same distinction ADR-0033 drew for a `NaN` height and ADR-0040 for a missing scan.

## Consequences

**Positive** — the third shape is storable, and it is storable in a form somebody can open in an
image viewer five years from now; the box rule is the same for all three shapes; the annotations
directory finally holds what the format reserved it for.

**Negative** — a project's annotation masks are files that `check_integrity` does not yet look at, so
a painted mask deleted behind the application's back is found when something tries to draw it rather
than when the project opens. Extending the integrity report to `annotations/` is a small task with a
decision behind it — what to do about an orphaned mask — and it is not this one.

**Neutral** — the layer loads every painted mask when it redraws. At a handful of masks per scan that
is a PNG read per annotation and nobody notices; at hundreds it is the first thing to cache.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A BLOB column | PROJECT_RULES §5, and a blob nothing but this application can read |
| `.npy` beside the database | Smaller and faster, and unopenable by anything an operator has |
| Name the file before inserting the row | An id nobody assigned, or a uuid nothing else in this project uses |
| Delete the file on undo | ADR-0040: forgetting and deleting are different decisions |
| Paint into the displayed scan | Edits the data being measured — the worst version of the feature |
| Draw a stored mask filled | Hides the pixels it describes (ADR-0064 §6) |

## Compliance

`tests/gui/test_brush_tool.py` asserts the file lands under `annotations/` with the row pointing at
it, that the box is derived from the painted pixels, that it reads back as what was painted, that an
empty stroke stores nothing and an empty mask is refused by the repository, that a missing file is a
**refusal** rather than an empty mask, that undo keeps the file and redo points at it again, and
that painting never touches the scan's pixmap.

## References

- PROJECT_RULES §5 — the rule this implements
- ADR-0042 — the same call for measurement tables
- ADR-0072 — the derived box, and the shape rule this extends
- ADR-0040 — what an undo leaves behind, and what a missing file means
