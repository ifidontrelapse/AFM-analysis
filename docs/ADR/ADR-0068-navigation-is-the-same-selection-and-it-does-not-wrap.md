# ADR-0068 — Navigation is the same selection, and it does not wrap

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M6-T08)
- **Affects:** `gui/viewmodels`, `gui/main_window`, `gui/panels/project_explorer` · M6

## Context

The workflow M6 has been assembling is *look at a scan, run it, read the numbers*. The real version
is doing that to forty scans in a row, and every one of them cost a trip back to the explorer, a
click, and a search for where you were.

## Decision

### 1. Navigation is `select_image` with a different id

Not a second mechanism. What is new is *which* id, taken from the open project's own order — and the
reason there is one place to ask is ADR-0057.

### 2. It does not wrap

Wrapping takes an operator from the fortieth scan back to the first without saying so, which in a
batch review means quietly starting again. The review that asks *"did I look at all of them?"* is
exactly the one that must not lie, so the actions go dead at the ends instead.

### 3. The status bar says which of how many

**"3 of 40"**, permanently. Half of navigating is knowing whether there is anywhere left to go, and
a label answers it without a trip to the list.

### 4. The explorer follows, with its signals blocked

A panel listing the images while a different one is on screen is a panel that lies. It sets its row
with `blockSignals`, because otherwise selecting the row asks the session for the selection it has
just announced — the loop M6-T05 already met on the measurements table.

### 5. The zoom does not survive the move

Every scan is fitted, because scans differ in size and a zoom held across a smaller one shows a
corner. *"Keep the view"* is a real feature for comparing two scans of one sample — with a control
and a name, not as a default nobody chose.

## Consequences

**Positive** — a forty-scan review is two shortcuts; the operator can always see where they are; the
explorer and the canvas cannot disagree about which scan is open.

**Negative** — the order is the project's own (the import order) and nothing sorts it, so an operator
who wants alphabetical order gets it by importing alphabetically. The actions are also disabled while
a job runs, like everything else that changes the selection, which means a batch review pauses while
a scan is analysed — correct, and worth noticing.

**Neutral** — the shortcuts are `Ctrl+←` / `Ctrl+→`. Nothing else in the window claims them, and the
graphics view uses bare arrows for scrolling.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Wrap around at the ends | A review that silently restarts is a review that lies |
| A "go to image N" box | A list already exists, three feet to the left |
| Let the explorer own the position | It is a panel; the selection has lived in the viewmodel since ADR-0057 |
| Keep the zoom across scans | Scans differ in size; a held zoom shows a corner of the next one |

## Compliance

`tests/gui/test_navigation.py` asserts the walk follows the project's order, that both ends refuse,
that the count is one-based and names the total, that the actions go dead at the ends and with no
project, and that the explorer's row follows **without** the selection being announced twice.

## References

- ADR-0057 — one place holds the selection
- ADR-0065 — the echo loop this avoids, met once already
