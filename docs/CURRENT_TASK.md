# CURRENT TASK

**ID:** `M7-T08`
**Title:** Undo/redo wired through every tool
**Milestone:** M7 — Annotation & metrology tools, eighth task
**Defect:** — · **ADR:** **ADR-0077**
**Branch:** `feat/m7-annotation-tools`
**Status:** **done 2026-08-17.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

Every tool M7 built already goes through the command stack — M7-T02 wrote that rule and M7-T03,
M7-T04, M7-T05 and M7-T07 kept it. So the task is **an audit**, and an audit's value is the gaps it
finds. It found three, and two of them were written down by the tasks that created them:

1. **Nothing tells the window that the *history* moved.** M7-T02 wired the Undo menu to
   `annotations_changed` and said in its own commit that this works *only* while every command
   mutates annotations; M7-T05's ruler ended that and added `rulers_changed` beside it. The next
   command that touches neither breaks it silently, and the debt names this task as the payer.
2. **Adopting forty detections costs forty undos.** `adopt_all_detections` runs one command per
   detection, so the one-click workflow ADR-0076 built needs `Ctrl+Z` forty times to take back.
3. **Undo can edit a scan nobody is looking at.** The stack is per project, annotations are per
   image, and `_history` reloads the *selected* image — so undoing an edit made on another scan
   removes a row the operator cannot see and the screen does not change.

---

## The decisions this task has to make

**1. The audit is the deliverable, and it is a table.**

Every mutating action reachable from the window is either an **edit** (on the stack) or **not an
edit** (and says why). Written down, because "is everything undoable?" is a question that gets
re-asked every milestone and answered by grepping.

| Action | On the stack | Why |
|---|---|---|
| Draw a box / outline / mask | yes | `AddAnnotation` (M7-T02, T03, T04) |
| Draw a ruler / profile line | yes | `AddRuler` (M7-T05) |
| Adopt a detection, adopt all | yes | `AddAnnotation` marked `from_detection` (M7-T07) |
| Rename, delete an annotation | yes | `UpdateAnnotation`, `RemoveAnnotation` (M7-T07) |
| Import images | no | Work, not an edit: it copies files (ADR-0043 vs ADR-0045) |
| Run an analysis | no | A run is a **record of what happened** (ADR-0042, ADR-0076 §1) |
| Remove an image | no | Deliberate, with a dialog instead (ADR-0045, ADR-0055) |
| Export a CSV | no | Produces a file outside the project's state |
| Change a setting | no | ADR-0047's scopes; a preference is not an edit to the work |
| Select an image, run, particle, box | no | Selection is not a mutation |

**2. The history announces itself.**

A `history_changed` signal on the session, emitted whenever a command runs and whenever the history
steps. The window listens to *that* for its Undo/Redo labels, not to `annotations_changed`. The stack
still knows nothing but order (ADR-0045) — what changed is that the *session* says so, because it is
the only thing that runs commands.

**3. One gesture is one undo.**

A `Composite` command holds several edits and reverses them in reverse order. `adopt_all_detections`
becomes one entry on the history labelled *"adopt 40 detection(s)"*. A child that fails takes back
the ones already done and re-raises, because `CommandStack.run` promises that a command which failed
is not on the history — half an adoption on the history is worse than none.

**4. Undo goes to the work it undoes.**

Each command says which image it edited. The session reads it off the command the stack hands back
and selects that image before redrawing. **The stack still does not learn what a command is:** it
never reads `image_id`; the session does, and only because it is the layer that owns the selection.
The alternative — leaving it — is an undo that silently edits a scan off screen, which is the
`M6-T08` standard restated: a review that lies about where it is is worse than one that is slow.

**5. A ruler cannot be deleted, and undo is not a delete button.**

`remove_ruler` exists on the repository with `AddRuler.undo` as its only caller. Removing the third
of five rulers needs a ruler *selection* on the canvas, which M7-T05 did not build. Filed as
**B-070** rather than smuggled in here: undo is for taking back what you just did, and using it as
the only way to remove anything is the workflow that loses the four rulers after it.

---

## Scope

**In scope**

1. `application/commands.py` — `Composite`, and `image_id` on the `Command` protocol
2. `gui/viewmodels/session.py` — `history_changed`, one funnel for running a command, undo/redo that
   follows the edit to its image
3. `gui/main_window.py` — the Undo/Redo labels from `history_changed`
4. **ADR-0077** — the audit, and the three gaps it closed
5. Tests: every tool puts exactly one entry on the history and announces it; adopt-all undoes in one
   step; a failing child leaves nothing behind; undo across images selects the scan it edited;
   closing a project empties the history and the menu says so
6. **B-070** filed

**Out of scope**

- **Deleting a ruler** — B-070, and it needs a canvas selection first
- **Persisting the history** — ADR-0045 refused it with an argument that still holds
- **Undoing an import, a run or an image removal** — each is a decision already made and recorded

---

## Definition of done

- [x] `history_changed` is the signal the window's Undo/Redo reads
- [x] Adopting every detection is one entry on the history
- [x] Undo selects the image whose work it took back
- [x] The audit table, in ADR-0077
- [x] `make check` green — 1282 tests, golden byte-identical, mypy unchanged at 6
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `Backlog.md` (B-070), `PROJECT_CONTEXT.md`
- [x] Commit: `M7-T08: one gesture, one undo — and the history says it moved`

---

## What it turned up

**Two of the three gaps had been written down by the tasks that made them.** M7-T02's own commit said
the Undo menu's signal held *only* while every command mutated annotations; M7-T05 found the first
command that did not and added a second signal beside the first. A third would have been the same
mistake again — which is what makes this the task that pays the debt rather than the one that adds to
it.

**Undo was being asked to do a delete button's job.** `remove_ruler` has had exactly one caller since
M7-T05, `AddRuler.undo`, so a ruler measured four edits ago cannot be removed at all — and the one
mechanism that reaches it also takes back the four edits after it. **B-070**, because deleting one
needs the canvas selection annotations got in M7-T07 and rulers never did.

**The cross-image gap was proved before it was fixed:** undoing a box drawn on the first scan while
looking at the second left the operator on the second, with nothing changed on screen.
