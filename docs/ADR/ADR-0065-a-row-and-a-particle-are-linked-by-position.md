# ADR-0065 — A row and a particle are linked by position

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M6-T05)
- **Affects:** `gui/panels`, `gui/viewmodels` · M6

## Context

M6's second exit criterion is *"selecting a table row highlights the particle, and vice versa"*. A
run that says a particle is 14 nm tall is a number; the same number next to the particle it came
from is evidence.

`measurements_for` has been on the port since M4-T05 with one reader — the CSV export.

## Decision

### 1. The link is a coordinate, not an index

The measurement table is a **subset** of the detections: a height that is not a number is discarded
(ADR-0033), so row *n* is not detection *n*. `x_px`/`y_px` are in the schema's core (ADR-0031) and on
every detection, so position is the link both sides carry — and the only one that survives a producer
which renumbers its rows.

**It has to be, because `particle_id` means two different things.** The baseline producer writes the
*blob's* index (`height.py`: `"particle_id": i`); the segmentation producer writes the index of the
row being appended (`sam2.py`: `res["particle_id"] = len(records)`), which renumbers after a drop.
One column, two meanings across producers — the defect class ADR-0031 removed from `radius_nm` — and
it is filed as **B-069** rather than fixed here, because changing what a producer writes moves stored
data and gets its own commit (ADR-0010).

### 2. The selection lives in the viewmodel, not in either widget

Both the table and the canvas *ask* for a particle; both are *told* by the session. Two widgets
talking to each other is what ADR-0057 removed, and "and vice versa" is exactly the case where it
would come back.

### 3. A click is a release that did not move

The view drags to pan (M5-T05), so a selection cannot be a press. Three pixels of tolerance, because
a mouse moves under a finger — without it, half an operator's clicks would be pans that selected
nothing.

### 4. The columns are the producer's own

Names and dtypes as stored (ADR-0031). A panel that renamed them for display would be a second
vocabulary, and whoever opened the exported CSV would find different words for the same measurement.
Floats are shown to four significant figures: that is a column width, not a rounding of the data.

### 5. A run that measured nothing says so

`detect` writes no table at all (ADR-0042). An empty grid with the right headers would claim it found
nothing rather than that nothing was asked for.

## Consequences

**Positive** — the criterion is met in both directions with one piece of state; a stored table is
readable without leaving the application; the coordinate link survives producers that number their
rows differently.

**Negative** — the match is a linear scan per row, done once per table: fine at hundreds of
particles, and the thing to index if a run ever produces tens of thousands. Two particles closer than
the tolerance would also match the same row, which cannot happen while detections are centres of
distinct blobs and would need a real answer if it ever does.

**Neutral** — no sorting and no filtering. The stored order is the producer's, and a sort is view
state nothing has asked for; when it arrives, the row→particle map has to be sorted with it.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Link by `particle_id` | It means two different things across producers — B-069 |
| Link by row index | The table is a subset; the indices diverge at the first discarded height |
| Let the table and the canvas talk to each other | ADR-0057, and "vice versa" is where it comes back |
| Select on mouse press | The press is a pan; every drag would select |
| Rename columns for display | A second vocabulary beside the exported CSV's |

## Compliance

`tests/gui/test_measurements_panel.py` asserts the table is the stored one with its own headers, that
a row selects the particle its coordinates name, that a click on the canvas selects the row, that the
selected outline is the thicker one, that a click on bare image clears both, and that a detect-only
run says it measured nothing.

## References

- ADR-0031 (one measurement schema) — the core columns the link uses
- ADR-0033 — the discarded rows that make the table a subset
- ADR-0042 — the stored table, and the mode that writes none
- ADR-0057 — why the selection is in the viewmodel
- **B-069** — `particle_id`'s two meanings
