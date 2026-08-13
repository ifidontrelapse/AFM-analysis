# ADR-0072 — An outline is stored beside the box

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** operator + agent (M7-T03)
- **Affects:** `core/entities`, `infrastructure/storage`, `application/commands`, `gui` · M7 · M8

## Context

M7-T02 refused the point tool because a point has no extent and **no reader**. The polygon is the
other side of that argument: M7's exit criterion asks for it, and a particle that is not a rectangle
is the ordinary case in this science — an operator outlining a cluster is drawing something a box
cannot express.

ADR-0044 wrote the condition for revisiting the shape decision itself:

> *"If M6 finds an operator drawing something a box cannot express, that shape then has a reader and
> this decision gets revisited."*

This is the first schema change since M4-T13, three milestones ago.

## Decision

### 1. The outline is stored **beside** the box, not instead of it

`annotations` gains one nullable `points` column, holding the vertices as JSON. `x1…y2` stay, and
for a polygon they are its **bounding box**. Therefore:

- every reader that consumes boxes keeps working, unchanged — M5-T04's confirmation, the layer
  M7-T01 draws, M8's detection dataset;
- **`points IS NULL` means *a box, drawn as a box***, which is what every row written before this is;
- **nothing migrates**, because "no outline" is exactly what the existing rows already mean.

### 2. The box is derived, never typed in

The repository computes it from the vertices, so a polygon and its bounding box cannot disagree. A
caller who hands in both is not offered the chance — the `box` argument is ignored when `points` is
given.

### 3. Fewer than three vertices is not an outline

Two are a line and one is the point ADR-0071 declined. The existing `CHECK (x2 > x1 AND y2 > y1)`
then refuses a degenerate outline for free, because its bounding box is degenerate too.

### 4. JSON in a column, not a second table

An outline is read and written **whole**, always, and never queried by vertex. A normalised
`annotation_points` table would buy ordering machinery and a join for that.

### 5. The undo stack carries the outline

`AddAnnotation` holds it, and `restore_annotation` writes it back. An undo that restored the box and
dropped the outline would silently redraw the operator's work as a rectangle — the quiet kind of
data loss this project keeps refusing.

### 6. The canvas draws the outline, and the sketch while it is made

A polygon drawn as its bounding box is a shape nobody made. And an outline the operator cannot see
until it is finished is one they draw twice, so the vertices appear as they are clicked; double-click
closes it, which is the gesture every annotation tool uses.

## Consequences

**Positive** — the shape an operator actually sees can be recorded; nothing that reads boxes had to
change; a project written by an older version opens and reads exactly as before.

**Negative** — an outline is not yet a mask, and M8's dataset builder will have to decide whether it
rasterises one. The polygon also has no vertex editing: fixing a misplaced click means drawing it
again until M7-T07. And a `points` column with JSON in it cannot be queried by SQLite — accepted,
because nothing queries a vertex.

**Neutral** — **the schema-history helper had to learn about columns.** v6 is the first migration in
this project that alters a table rather than adding one, and `revert_to` — "drop the tables a later
step created" — left the `points` column behind, so re-running the step answered *duplicate column
name*. It now undoes columns as well, from a second map with the same guard over it.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Replace the box with a shape column | Every box reader in the project would have to change, for a feature none of them use |
| A normalised `annotation_points` table | Ordering machinery and a join for something read and written whole |
| Store the outline and compute the box on read | Every reader pays for the derivation, forever, instead of the one writer |
| Let the caller pass both | Two sources of truth for one rectangle |
| Draw a polygon as its bounding box | A shape nobody made — the substitution this project keeps refusing |

## Compliance

`tests/gui/test_polygon_tool.py` asserts the outline is kept and the box derived, that a box
annotation still has **no** outline, that it survives the process, that two vertices are refused in
the session *and* in the repository, that **redo puts the polygon back rather than its box**, that a
polygon draws as a polygon and a box as a box, and that the sketch is visible while it grows.
`tests/integration/schema_history.py` covers the revert, with a guard over both maps.

## References

- ADR-0044 — one shape, and the condition §1 satisfies
- ADR-0071 — the point tool, refused for the reason this one is not
- ADR-0039 — the migration mechanism, and "never edit a step that has shipped"
- ADR-0045 / M4-T08 — the undo §5 keeps honest
