# CURRENT TASK

**ID:** `M5-T04`
**Title:** The project explorer, and the confirmation ADR-0044 asked for
**Milestone:** M5 — GUI shell, fourth task
**Defect:** — · **ADR:** **ADR-0055**
**Branch:** `feat/m5-gui-shell`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M5-T02 left a dock saying "the project explorer arrives in M5-T04". More importantly, **ADR-0044
ended on an obligation addressed to this task by name**: annotations cascade when an image is
removed, and

> *`annotations_for` exists to be counted **before** the deletion, by a confirmation dialog that
> can say "this image has 12 annotations".*

Until now nothing could remove an image, so the obligation had nowhere to land. This is the first
panel that can, so it is the task that pays it.

---

## The decisions this task has to make

**1. What does the confirmation actually say?** The **count**, and what it means.

Not "Are you sure?" — which is a dialog people click through — but *"scan_01.spm has 12 annotations.
Removing the image deletes them. The file itself stays in `images/`."* Three facts, because all
three are non-obvious: annotations are hand work that cannot be recomputed (ADR-0044), the file is
**not** deleted (ADR-0040's `remove_image` leaves it), and what remains becomes an untracked file
the integrity check will report.

An image with **no** annotations is removed without a dialog. A confirmation that always appears is
one nobody reads by the third time, and then the one that mattered is clicked through too.

**2. Does the panel talk to the repository or to a viewmodel?** To the container, as `MainWindow`
does. M5-T06 owns the viewmodel layer, and inventing half of one here — for a panel with a list and
two actions — is the abstraction this project keeps declining.

What *does* change: `Nanoscope.repository` is typed as the **port**, not the SQLite class. The GUI
may not import `infrastructure` (Architecture §3.2), and a container that hands out a concrete
adapter is a container that makes the rule unenforceable.

**3. What does the panel show?** Name, modality, scale — and **which files are missing**, marked
from the integrity report the container already carries. A dock that lists an image whose file is
gone, without saying so, is a panel that lies quietly.

**4. Is removal undoable?** No, and that is ADR-0045's rule rather than an omission: the command
stack holds annotation edits, and removing an image is guarded by the confirmation in decision 1
instead. Making it undoable means holding a deleted image's rows in memory to reinsert them, which
is a bigger promise than a dialog.

---

## Scope

**In scope**

1. `gui/panels/project_explorer.py` — the list, the selection signal, the remove action
2. The confirmation, counting annotations before it asks
3. `MainWindow` wiring: the dock's placeholder replaced, the panel populated on open and cleared
   on close
4. `Nanoscope.repository` typed as `ProjectRepository`
5. **ADR-0055** — what the confirmation says and when it appears, the port in the container
6. Tests: population, a missing file marked, selection emitted, the confirmation counting
   correctly, cancelling changing nothing, and removal without annotations not asking at all

**Out of scope**

- **The image viewer** — M5-T05 consumes the selection signal this task emits
- **Importing images from the panel** — a file dialog and a job; M5-T07's shape
- **A viewmodel layer** — M5-T06

---

## Definition of done

- [x] A panel listing the project's images, with missing files marked
- [x] The confirmation says the count, the consequence and what survives
- [x] No dialog when there is nothing to lose
- [x] ADR-0055
- [x] Headless tests, including cancelled and unguarded removals
- [x] `make check` green — 902 tests, golden byte-identical
- [x] Docs, the ADR index
- [x] Commit: `M5-T04: a panel that counts what a deletion would cost`
