# ADR-0055 — A confirmation that counts what it would cost

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M5-T04)
- **Affects:** `gui/panels`, `app/container` · M5 · M5-T05's viewer

## Context

ADR-0044 decided that annotations cascade when an image is removed — a box pointing at an image the
project no longer knows about is not an annotation of anything — and then wrote an obligation
addressed to a task that did not exist yet:

> *`remove_image` … now discards hand work that cannot be recreated … **`annotations_for` exists to
> be counted before the deletion**, by a confirmation dialog that can say "this image has 12
> annotations".*

Nothing could remove an image until this panel. This is the task that pays it.

## Decision

### 1. The dialog says the count, and what survives

Not "Are you sure?" — the dialog people click through — but three facts, because all three are
non-obvious:

- **`scan_01.spm` has 12 annotation(s)**, and removing the image deletes them;
- **they cannot be recomputed**, which is the whole reason ADR-0044 made them rows;
- **the file itself stays** in `images/`, and will be reported as untracked afterwards.

The third fact is the one an operator gets wrong in the other direction: a dialog that says only
"delete?" leaves them believing their scan was deleted, and the scan is still there.

### 2. No dialog when there is nothing to lose

An image with no annotations is removed without asking.

A confirmation that always appears is one nobody reads by the third time — and then the one that
mattered is clicked through as well. The guard is worth having *because* it is rare, and making it
rare is the only way to keep it worth reading.

### 3. `Nanoscope.repository` is typed as the port

Not as `SqliteProjectRepository`. `gui/` may not import `infrastructure` (Architecture §3.2), and a
container that hands out a concrete adapter makes the rule unenforceable — mypy would happily infer
the SQLite class into every panel, and the guard that checks *imports* would never notice.

The first panel is where this becomes real, so it changes here rather than in a tidy-up later.

### 4. The panel talks to the container, not to a viewmodel

M5-T06 owns the viewmodel layer. Inventing half of one for a list and two actions is the
abstraction this project has declined at every previous opportunity (M2-T08's ports, ADR-0041's use
cases, ADR-0046's autosave, ADR-0052's container).

When M5-T05's viewer needs the same selection, there will be two consumers and a reason.

### 5. A missing file is marked, not hidden

The integrity report is already in hand — the container reads it on open (ADR-0052) — so the row
says *"— file missing"* and greys itself. **A panel that lists an image whose file is gone without
saying so is a panel that lies quietly**, and the honest alternative to showing it is not hiding it
but explaining it: the row is kept on purpose (ADR-0040).

### 6. Removal is not undoable, and that is ADR-0045's rule

The command stack holds annotation edits. Making a removal undoable means holding a deleted image's
rows — and its annotations, and its runs — in memory to reinsert them, which is a larger promise
than the confirmation in §1. The dialog is the guard; undo is for edits.

## Consequences

**Positive**

- ADR-0044's obligation is discharged where it was addressed, with the count it asked for.
- The dialog appears rarely enough to be read, and says the two things an operator would otherwise
  guess wrong.
- The GUI cannot reach an infrastructure type by inference any more.
- M5-T05 has a selection signal to consume, emitted by something that already exists.

**Negative**

- The panel rebuilds itself from the repository after a removal rather than editing its list. That
  is a query per removal, and the reason is worth the cost: the integrity report is part of what is
  displayed, and the removed image's file has just become untracked.
- Removal without undo will surprise somebody who has just discovered undo works for annotations.
  §6 is that surprise, written down.

**Neutral**

- The panel is a `QTreeWidget` rather than a model/view pair. At two columns and no sorting, a
  model would be three classes to do what a list does.

## Alternatives considered

| Alternative | Why not |
|---|---|
| "Are you sure?" | The dialog people click through; it carries none of the three facts |
| Always confirm, even with nothing to lose | Trains the operator to dismiss the one that mattered |
| Delete the file too | ADR-0040: forgetting a scan and deleting it are different decisions |
| Hide images whose file is missing | The row is kept on purpose; hiding it makes the project look smaller than it is |
| Make removal undoable | Holding a deleted image's rows, runs and annotations in memory — a bigger promise than a dialog |
| A viewmodel now | Half an abstraction for one list, before there is a second consumer |

## Compliance

- `tests/gui/test_project_explorer.py` asserts the dialog carries the count, the "cannot be
  recomputed" sentence and the file's path; that cancelling changes nothing; that an image with no
  annotations is removed **without** a dialog; that the file survives; and that the panel refreshes
  and the file is then reported untracked.
- `Nanoscope.repository` is `ProjectRepository | None`; no module under `gui/` imports
  `infrastructure`.

## References

- ADR-0044 (an annotation is a row) — the obligation this discharges
- ADR-0040 (the repository reports and does not reconcile) — why the file stays and the row is kept
- ADR-0045 (undo is a session) — §6
- ADR-0052 / ADR-0053 — the container and the window this panel sits in
- `docs/Architecture.md` §2.3, §3.2
