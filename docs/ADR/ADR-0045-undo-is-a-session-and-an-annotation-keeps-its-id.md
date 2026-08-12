# ADR-0045 — Undo is a session, and an annotation keeps its id

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T08)
- **Affects:** `application/commands`, `core/ports`, `infrastructure/storage` · M4 · M6's editor

## Context

M4-T07 gave the operator something to get wrong — annotations are hand-made and cannot be
recomputed — and the same task made `remove_image` capable of taking a dozen of them with it. M4's
exit criterion is therefore a safety property rather than a feature: *"undo/redo proven on at least
one mutating use case"*.

Architecture §4.5 fixed the shape before there was anything to undo: a command stack in
`application/commands.py`, every mutating action a command with `do()` and `undo()`, and a GUI that
dispatches rather than mutates.

## Decision

### 1. The stack knows nothing but order

`CommandStack.run(command)` calls `do()`, keeps the object, and clears the redo list. It never
learns what a command *is* — no repository, no annotation, no project.

A stack that knows about annotations is a stack that has to be extended for measurements, and then
for settings. Forty lines that will never need to change, against a class that grows with every
table.

Redo is `do()` again, not a third method: a command that cannot repeat itself is a command that
was written to be undone once.

### 2. Undo is a session, and persisting it is a promise not made

The history dies with the process. It is not written to the project.

Replaying edits across sessions means replaying them against a directory that may have changed on
disk in between — and ADR-0040 already established that this application does not assume it is the
only thing touching a project. **An undo history that can be silently wrong is worse than one that
is honestly short.**

The durable record of what happened is M4-T14's log, which is a different thing: history, not
reversibility.

### 3. An annotation keeps its id through undo and redo

`restore_annotation(annotation)` on the repository puts a deleted row back **as itself** — same id,
same timestamps — and it is a separate operation from `add_annotation`, because creating a box and
undoing its deletion are different acts and only one of them may choose an id.

**This reversed the plan, and the test that reversed it is in the suite.** The first design let a
redo insert a fresh row with a new id, on the grounds that an `id=` parameter would be a back door.
Then `test_a_sequence_undoes_and_redoes_whole` was written: add a box, edit the box, undo twice,
redo twice. With a new id on the redo of the add, the redo of the edit points at a row that does
not exist — so **undo would have been one command deep in practice**, which is not undo.

Reclaiming an id is safe under a LIFO stack: anything created after a deletion is undone before it,
so the id is free by the time it is wanted. Outside that discipline the database's own `UNIQUE`
refuses the collision, which is the right answer to restoring something twice.

### 4. A failing undo propagates, and the history does not move

The stack assumes it is the only writer. When it is wrong — a cascade, a script, a second window —
the inverse operation can fail. It raises, and the pointer stays where it was.

Swallowing the error would leave the history describing a state that never existed, which is worse
than an error message: the *next* undo would then be wrong too, silently.

### 5. Three commands, all over annotations

Add, update, remove. They are the only hand-made data in the project, which makes them the only
edits whose loss is unrecoverable.

Not `import_images`: undoing it means deleting files the operator asked to be copied, which is the
deletion ADR-0040 refused to make automatically. Not `run_analysis`: a stored result is derived, and
the inverse of computing something is not an edit.

### 6. No coalescing

Ten small drags stay ten undos. Merging them needs a rule about what "the same edit" means, and the
only honest source for that rule is watching somebody use an editor that M6 has not built.

## Consequences

**Positive**

- M4's exit criterion is met, and met against a real database rather than a stack of mocks.
- Sequences of edits are reversible as sequences, not one step at a time.
- The stack is small enough to read in one sitting and will not grow when new tables arrive.
- A GUI can now be legal: it dispatches commands, and Architecture §2.3's "no business logic in a
  widget" has something to dispatch *to*.

**Negative**

- Closing a project loses the history, and an operator who expects undo after reopening will be
  surprised. Deliberate, per §2, and the surprise is a documented limit rather than a wrong answer.
- `restore_annotation` is a repository method that only undo has a use for. Its docstring says so,
  and its safety depends on the LIFO discipline stated in §3.
- Nothing coalesces, so a drag implemented as ten updates would fill the history. That is a signal
  for M6 to issue one command per gesture, not for the stack to guess.

**Neutral**

- The stack is not thread-safe, on purpose: edits are a user's sequence, and jobs (M4-T06) run
  *work* rather than edits.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A stack that knows the repository and mutates directly | Every new table extends the stack; commands exist so it does not have to |
| Persisting the undo history in the project | Replaying edits against a directory that may have changed — an undo that can be wrong |
| A fresh id on every redo | Breaks any command stacked above the annotation, making undo one step deep (§3) |
| `add_annotation(..., annotation_id=)` | The same capability with a name that invites using it for creation |
| Snapshot the whole project and diff | Simple to write, unbounded to store, and useless for telling an operator *what* it will take back |
| Swallow a failing undo and move on | The next undo is then wrong too, silently |
| Coalescing consecutive edits by a timer | A rule invented before the editor that would produce the edits |

## Compliance

- `tests/unit/test_commands.py` covers the stack alone: ordering, the redo list cleared by a new
  command, an empty stack answering `None`, labels for the menu, a failed `do()` staying off the
  history, and a failing `undo()` leaving it in place.
- `tests/integration/test_undo.py` proves the **database** goes back, including a two-command
  sequence undone and redone whole and an id that survives the round trip.
- One test deletes a row behind the stack's back and asserts the undo raises and the history stands
  still.
- `UpdateAnnotation` captures the previous values in `do()`; a test with two consecutive edits fails
  if it ever reads them at `undo()` time instead.

## References

- `docs/Architecture.md` §4.5 — the shape this implements
- ADR-0044 (an annotation is a row) — the data this makes reversible
- ADR-0040 (the repository reports and does not reconcile) — why §2 does not persist a history
- ADR-0043 (cancellation is asked for, not forced) — jobs are work; commands are edits
- `docs/Roadmap.md` M4 — "undo/redo proven on at least one mutating use case"
