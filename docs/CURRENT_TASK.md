# CURRENT TASK

**ID:** `M7-T03`
**Title:** A polygon is a box that kept its outline
**Milestone:** M7 — Annotation & metrology tools, third task
**Defect:** — · **ADR:** **ADR-0072**
**Branch:** `feat/m7-annotation-tools`
**Status:** **done 2026-08-14.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M7-T02 shipped the box tool and refused the point tool, because a point has no extent and **no
reader**. The polygon is the other side of that argument: M7's own exit criterion says *"all seven
annotation/measurement tools usable and persisted"*, and a particle that is not a rectangle is the
ordinary case in this science — an operator outlining a cluster is drawing something a box cannot
express.

ADR-0044 wrote the condition for revisiting the shape decision, and this is it:

> *"If M6 finds an operator drawing something a box cannot express, that shape then has a reader and
> this decision gets revisited."*

So this task changes the schema, which makes it the first one in three milestones to do so.

---

## The decisions this task has to make

**1. What is stored?** The outline, **beside** the box — not instead of it.

`annotations` gains a nullable `points` column holding the vertices as JSON; the existing `x1…y2`
stay, and for a polygon they are its **bounding box**. So:

- every reader that consumes boxes keeps working, unchanged — M5-T04's confirmation, M8's detection
  dataset, and the layer M7-T01 draws;
- `points IS NULL` means *a box, drawn as a box*, which is what every row written so far is;
- nothing has to migrate data, because the column is added empty and means "no outline".

A separate `annotation_points` table would be the normalised answer and would buy nothing: the
outline is read and written whole, always, and never queried by vertex.

**2. Does the box stay authoritative?** For anything that wants a box, yes — and it is **derived**,
not typed in.

The repository computes it from the vertices, so a polygon and its box cannot disagree. A caller who
hands in both is not offered the chance.

**3. What refuses what?** Fewer than three vertices is not an outline.

Two points are a line and one is the point M7-T02 declined; the `CHECK` on `x2 > x1 AND y2 > y1`
then also refuses a degenerate polygon for free, because its bounding box is degenerate too.

**4. How is it drawn?** Click to add a vertex, double-click to close.

The gesture every SPM and annotation tool uses. An outline in progress is visible while it is being
made, because a polygon the operator cannot see until it is finished is a polygon they draw twice.

**5. What does the layer draw?** The outline when there is one, the box when there is not.

Same colours, same source distinction (ADR-0070); a polygon drawn as its bounding box would be a
shape nobody made, which is the substitution this project keeps refusing.

---

## Scope

**In scope**

1. `core/entities/project.py` — `Annotation.points`
2. `infrastructure/storage/database.py` — schema **v6**, one added column
3. `infrastructure/storage/project_repository.py` — write, read back, restore, and the derived box
4. `application/commands.py` — `AddAnnotation` carries an outline, so undo/redo does too
5. `gui/` — the polygon tool, the outline in progress, and the layer drawing polygons
6. **ADR-0072** — the outline beside the box, and why the box stays
7. Tests: the migration, the round trip, the derived box, a refused two-vertex outline, undo/redo,
   and the drawing gesture

**Out of scope**

- **Editing a vertex after the fact** — M7-T07
- **The brush** — M7-T04, which needs a mask and not an outline
- **Converting a polygon to a mask** — M8's dataset builder decides what it consumes

---

## Definition of done

- [x] A polygon round-trips: outline stored, bounding box derived, `points IS NULL` still a box
- [x] Old projects open and their boxes read unchanged
- [x] Undo and redo carry the outline
- [x] The canvas draws the outline while it is being made, and after
- [x] ADR-0072 + the ADR index
- [x] `make check` green — 1192 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, `ProjectFormat.md`
- [x] Commit: `M7-T03: a polygon is a box that kept its outline`

---

## What it turned up

**v6 is the first migration in this project that alters a table rather than adding one**, and the
schema-history helper was built on the assumption that it never would be. `revert_to` drops "the
tables a later step created", so reverting to v3 left `annotations` carrying a v6 column, and
re-running the step answered `duplicate column name: points` — in `test_project_settings.py`, a test
about something else entirely, which is exactly the failure that helper exists to prevent. It now
undoes columns as well, from a second map with the same guard over it.
