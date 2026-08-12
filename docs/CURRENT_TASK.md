# CURRENT TASK

**ID:** `M4-T08`
**Title:** Undo/redo: a stack that owns nothing but the order of things
**Milestone:** M4 — Application layer, eighth task
**Defect:** — (W1: no application layer exists) · **ADR:** **ADR-0045**
**Branch:** `feat/m4-application-layer` — M4 changes no scientific output (PROJECT_RULES §7)
**Status:** **done 2026-08-12.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M4-T07 gave the operator something to get wrong. An annotation is the one thing in a project that
cannot be recomputed, and the previous task also made `remove_image` capable of destroying a dozen
of them — so the milestone's exit criterion is now a safety property rather than a feature:

> *Undo/redo proven on at least one mutating use case.*

Architecture §4.5 fixed the shape before there was anything to undo: *a command stack in
`application/commands.py`; every mutating user action is a command with `do()` and `undo()`; the
GUI dispatches commands and never mutates state directly.*

---

## The decisions this task has to make

**1. What does the stack know?** Nothing but order. `CommandStack.run(command)` calls `do()`,
keeps the object, and clears the redo list. It never learns what a command *is* — no repository, no
annotation, no project. A stack that knows about annotations is a stack that has to be extended for
measurements, and then for settings.

**2. Is undo persistent?** **No, and this is a promise not to make one.** The stack lives as long
as the session does.

Persisting it means replaying edits against a project that may have changed on disk in between —
and ADR-0040 already established that this application does not assume it is the only thing
touching a project directory. An undo history that can be wrong is worse than one that is honestly
short. The durable record of what happened is M4-T14's log, which is a different thing: history,
not reversibility.

**3. Redo of an "add" produces a new id — who says so?** The command does, out loud.

Undoing an add deletes the row; redoing it inserts a new one, and SQLite assigns a fresh id. The
alternative — reinserting with the old id — means an `add_annotation(..., id=)` back door on the
repository, which exists solely to lie about identity. So the command exposes the *current*
annotation and the rule is stated: **nothing may cache an annotation id across an undo.** M6 reads
it from the command after each operation.

**4. What happens if `undo()` fails?** It propagates, and the stack does not move.

The stack assumes it is the only writer. If something changed the project behind its back — another
window, a script, `remove_image` cascading — the inverse operation can fail, and the honest
response is to say so and leave the history where it was, rather than to swallow the error and
leave the pointer describing a state that never existed.

**5. Which commands ship?** The three that mutate an annotation: add, update, remove. They are the
only hand-made data in the project, which makes them the only edits whose loss is unrecoverable —
and the exit criterion asks for one.

Not `ImportImages` (undoing it means deleting the operator's copied files, which is the deletion
ADR-0040 refused to do automatically), and not `run_analysis` (a stored result is derived; the
inverse of running it is deleting a run, which nobody has asked for).

**6. Does the stack coalesce?** No. Ten small drags stay ten undos. Merging them needs a rule about
what "the same edit" means, and the only honest source for that rule is watching somebody use the
editor, which M6 has not built.

---

## Scope

**In scope**

1. `application/commands.py` — the `Command` protocol, `CommandStack`
2. `AddAnnotation`, `UpdateAnnotation`, `RemoveAnnotation` commands over the port
3. **ADR-0045** — the stack knows nothing, undo is a session, the id changes, failures propagate
4. Tests: do/undo/redo round trips, the redo list cleared by a new command, labels for the menu,
   an empty stack, a failing undo leaving the history intact, and an integration test proving the
   database really went back

**Out of scope**

- **Persisting the history** — decision 2
- **Coalescing** — decision 6
- **Commands for imports and analyses** — decision 5
- **A GUI** — M5/M6. This is the layer that makes theirs legal

---

## Expected blast radius

- **Zero golden differences.** No numerical code is imported
- One new application module, one ADR, one unit test file, a few integration tests
- No new dependency

---

## Definition of done

- [x] `CommandStack` with `run` / `undo` / `redo`, and labels for a menu
- [x] Three annotation commands, driven through the port
- [x] ADR-0045
- [x] Tests, including a failing undo and a real database going back
- [x] `make check` green — 658 tests, golden byte-identical
- [x] `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, `Roadmap.md`, ADR index
- [x] Commit: `M4-T08: undo is a session, and it says so`

---

## What it turned up

**Decision 3 was wrong, and its own test proved it.** The plan said a redo should insert a fresh
row, because an `id=` parameter looked like a back door whose only purpose was to lie about
identity. Then the sequence test was written — add a box, edit the box, undo twice, redo twice —
and with a new id on the redo of the add, the redo of the edit points at a row that no longer
exists. **Undo would have been one command deep in practice, which is not undo.**

`restore_annotation` is the answer: a deleted row goes back *as itself*, and it is a separate
operation from `add_annotation` because creating a box and undoing its deletion are different acts.
Reclaiming an id is safe under a LIFO stack — anything created after a deletion is undone before
it — and outside that discipline the database's `UNIQUE` refuses, which is the right answer to
restoring something twice.

**Capturing the previous values at `do()` time is load-bearing, not an optimisation.** A command
that looked them up when *undone* would restore the second edit's starting point instead of its
own. Two consecutive edits, undone in order, is the test that catches it.

---

## Notes

The golden held for the eighth time. **M4-T09** takes autosave, and starts from an honest question:
every write already commits, so what is left for it to save?
