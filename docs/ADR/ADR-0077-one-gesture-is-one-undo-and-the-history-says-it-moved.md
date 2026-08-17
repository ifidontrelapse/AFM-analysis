# ADR-0077 — One gesture is one undo, and the history says it moved

- **Status:** Accepted
- **Date:** 2026-08-17
- **Deciders:** operator + agent (M7-T08)
- **Affects:** `application/commands.py`, `gui/viewmodels`, `gui/main_window.py` · M7 · M8

## Context

Every tool since M7-T02 already goes through the command stack, so the task *"undo/redo wired through
every tool"* is an **audit**. An audit's value is the gaps it finds, and two of the three were written
down by the tasks that created them: M7-T02 recorded that the window learned about the history from
`annotations_changed` and that this held **only** while every command mutated annotations, and M7-T05
found the first command that did not.

## Decision

### 1. The audit, written down

Every mutating action reachable from the window is either an **edit** — on the stack — or **not an
edit**, with the reason. The table is the deliverable, because *"is everything undoable?"* is a
question that otherwise gets re-answered by grepping once a milestone.

| Action | On the stack | Why |
|---|---|---|
| Draw a box / outline / painted mask | yes | `AddAnnotation` (M7-T02, M7-T03, M7-T04) |
| Draw a ruler / profile line | yes | `AddRuler` (M7-T05) |
| Adopt a detection, adopt all | yes | `AddAnnotation` marked `from_detection` (M7-T07) |
| Rename, delete an annotation | yes | `UpdateAnnotation`, `RemoveAnnotation` (M7-T07) |
| Import images | no | Work, not an edit: it copies files into the project (ADR-0043) |
| Run an analysis | no | A run is a **record of what happened** (ADR-0042, ADR-0076 §1) |
| Remove an image | no | Deliberate, with a confirmation instead (ADR-0045, ADR-0055) |
| Export a CSV | no | Produces a file outside the project's state (ADR-0048) |
| Change a setting | no | A preference is not an edit to the work (ADR-0047) |
| Select an image, run, particle, annotation | no | A selection is not a mutation |

### 2. The history announces itself

`SessionViewModel.history_changed` is emitted when a command runs, when the history steps, and when a
project is opened or closed — which clears the stack (ADR-0045). The window's Undo/Redo labels listen
to **that**.

Piggybacking on the layer signals worked twice and would have failed on the third command: M7-T02
could rely on `annotations_changed`, M7-T05's ruler forced `rulers_changed` beside it, and a third
would have been the same mistake again. **The stack still knows nothing but order** — what changed is
that the *session* says so, because it is the only thing that runs commands.

### 3. One gesture is one undo

`Composite` holds several commands, does them in order and undoes them in reverse.
`adopt_all_detections` is one entry on the history labelled *"adopt 40 detection(s)"*, because
ADR-0076 §3 made adoption one click and forty `Ctrl+Z` to reverse one click is a workflow nobody uses
— what they do instead is close the project.

A child that fails takes back the ones already done and re-raises, because `CommandStack.run` promises
that a command which raised is not on the history: half a batch on the history is an edit nothing can
reverse.

### 4. Undo goes to the work it undoes

The history is per project and the annotation layer is per image, so undoing an edit made on another
scan removed a row nobody could see and left the window unchanged. Each command now says which image
it edited; the session reads it off the command the stack hands back and selects that scan first.

**The stack does not read it.** ADR-0045's rule is intact — the session reads `image_id`, and only
because it is the layer that owns the selection. `UpdateAnnotation` and `RemoveAnnotation` take it
from the row they captured rather than from an argument, so it cannot disagree with the annotation it
describes.

### 5. Undo is not a delete button

`remove_ruler` exists on the repository with `AddRuler.undo` as its only caller, so the third of five
rulers cannot be removed at all — deleting one needs a ruler *selection* on the canvas, which M7-T05
did not build. Filed as **B-070** rather than smuggled in here: undo takes back what you just did, and
using it as the only way to remove anything costs the four edits made after the one you wanted gone.

## Consequences

**Positive** — the Undo menu is correct for a command that touches something no existing signal
describes, which is the next tool's problem solved before it is written; one click is one undo; and
undoing across scans shows the work it changed.

**Negative** — undo now moves the selection, which is a side effect of a menu item. It is the lesser
of two: the alternative is an edit applied to a scan off screen. The zoom does not survive the move,
for the reason M6-T08 gave.

**Neutral** — a ruler still cannot be deleted (B-070), and the history is still a session (ADR-0045).

## Alternatives considered

| Alternative | Why not |
|---|---|
| A third layer signal for the next command | The mistake M7-T02 wrote down, made twice more |
| Teach the stack which layer to reload | The stack knows nothing but order — that is what keeps it from growing per table (ADR-0045) |
| One history per image | Two operators' worth of bookkeeping for one project, and a cross-image batch belongs to neither |
| Leave adopt-all as N commands | Forty `Ctrl+Z` for one click; the undo exists but nobody can afford it |
| Report the label instead of selecting the scan | Says what happened, still leaves an edit applied where nobody can see it |
| Persist the history | ADR-0045 refused it with an argument that still holds |

## Compliance

`tests/gui/test_undo_history.py` asserts that **each of the five tools** is one entry on the history
and one `history_changed`, that the window's Undo item names what it would take back and follows a
command that touches no annotation, that closing a project empties both, that adopting every detection
undoes in one step and redoes with the **same row ids**, and that undo selects the scan whose work it
took back — while an edit on the selected scan does **not** reload it. `tests/unit/test_commands.py`
covers `Composite` against commands that only record being called: reverse order, the rollback of a
failed child, and where its `image_id` comes from.

## References

- ADR-0045 — the stack knows nothing but order; undo is a session
- ADR-0071 / ADR-0074 — the two tasks that wrote this task's findings down
- ADR-0076 — adoption as one click, which this makes one undo
- ADR-0055 — when a confirmation is worth having, and when undo is enough
