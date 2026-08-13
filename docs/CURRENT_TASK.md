# CURRENT TASK

**ID:** `M6-T08`
**Title:** Moving through a project's scans without going back to the list
**Milestone:** M6 — Analysis workflow in the GUI, eighth task
**Defect:** — · **ADR:** **ADR-0068**
**Branch:** `feat/m6-analysis-workflow`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

The workflow M6 has been assembling is *look at a scan, run it, read the numbers* — and the real
version of that is doing it to forty scans in a row. Every one of them currently costs a trip back to
the explorer, a click on the next row, and a search for where you were.

It is also the last task before M6-T09, which has to prove results survive a restart: proving it is
easier when moving between images is one keystroke.

---

## The decisions this task has to make

**1. Where does the selection live?** In the viewmodel, as it already does.

Navigation is `select_image` with a different id — not a second mechanism. What is new is *which* id,
and that comes from the open project's own order (ADR-0057, and the reason there is one place to ask).

**2. Does it wrap?** No. The ends are the ends.

Wrapping takes an operator from the fortieth scan back to the first without saying so, which in a
batch review means quietly starting again — and the review that is *"did I look at all of them?"* is
exactly the one that must not lie. The actions go dead instead.

**3. How does an operator know where they are?** A count in the status bar: **"3 of 40"**.

Half of navigating is knowing whether there is anywhere left to go. A permanent label costs one
widget and answers it without a trip to the list.

**4. Does the explorer follow?** Yes, and without echoing.

A panel listing the images while a different one is on screen is a panel that lies. It sets its row
with its signals blocked, because otherwise selecting the row asks the session for the selection it
just announced — the loop M6-T05 already met once, on the measurements table.

**5. Does the zoom survive the move?** No, and that is a deliberate absence.

Every scan gets fitted, because scans differ in size and a zoom held across a smaller one shows a
corner. A *"keep the view"* toggle is a real feature for comparing two scans of the same sample, and
it is a feature, with a control and a name — not a default.

---

## Scope

**In scope**

1. `gui/viewmodels/session.py` — `select_next`, `select_previous`, and the position in the project
2. `MainWindow` — the two actions, their shortcuts, and the "3 of 40" label
3. `gui/panels/project_explorer.py` — the row follows the session
4. **ADR-0068** — one selection mechanism, no wrapping, and where the operator is
5. Tests: the order is the project's, the ends disable the actions, the explorer follows without
   echoing, and the count is right

**Out of scope**

- **Keeping the zoom across images** — decision 5
- **Filtering or sorting the list** — the project's order is the import order, and a sort is view
  state nothing has asked for
- **Running a batch across images** — analysing forty scans unattended is a job that reports per
  scan, and it is not this task

---

## Definition of done

- [x] Next and previous move through the project's order, and stop at the ends
- [x] The status bar says which scan of how many
- [x] The explorer's row follows, without a loop
- [x] ADR-0068 + the ADR index
- [x] `make check` green — 1145 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M6-T08: moving through a project's scans without going back to the list`

---

## What it turned up

**`next_action` was enabled with no project open.** The window set every action's state from the
session's signals and never from its own initial state, so between construction and the first signal
each one sat in whatever Qt's default was. A test asked the obvious question and got the wrong
answer; the window now starts in the state the session implies.
