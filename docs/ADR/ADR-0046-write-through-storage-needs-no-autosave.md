# ADR-0046 — Write-through storage needs no autosave, and saying so is the deliverable

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T09)
- **Affects:** `application` · M4-T09 · M5's GUI

## Context

Architecture §4.5 scheduled autosave before there was any storage to autosave: *"a service that
persists dirty state on a timer and on well-defined events. Enabled by default."*

That sentence assumes there is dirty state. Eight tasks of storage later there is not, and the
first thing this task had to do was check rather than build.

## Decision

**No autosave service. Storage is write-through, and the claim ships as tests instead of as a
timer.**

Every mutating method on the repository commits before it returns — `add_image`, `import_image`,
`remove_image`, `save_analysis`, `add_annotation`, `restore_annotation`, `update_annotation`,
`remove_annotation`. There is no buffer, no dirty flag, and no "unsaved changes" state anywhere in
the application layer:

- ADR-0043's threading work made every write a short self-contained transaction, taken under one
  lock, so a write is durable the moment the call returns;
- ADR-0045 decided the one piece of deliberate in-memory state — the undo history — is **not**
  persisted, and gave the reason;
- ADR-0042 writes the measurement table to its file inside the same call that records the run.

An autosave service would therefore be a timer that flushes nothing. That is not merely useless: it
would create the *impression* of protection where the protection actually lives, and the first
person debugging a lost edit would inspect the timer instead of the write path.

**What ships instead** is `tests/integration/test_durability.py`, which proves the property rather
than asserting it: repositories abandoned without `close()`, a second connection reading a row
while the writer is still alive, every write path checked individually, and a **separate process
killed with `SIGKILL`** between writes with its rows intact afterwards.

**What would reverse this decision**, named so it has a trigger rather than a mood:

1. **View state that lives only in the GUI** — M5's zoom, selection, panel layout. That is
   "remember what I was doing", it belongs in settings, and M4-T10 is the next task.
2. **Any write path that batches.** If a future change buffers writes for speed, the durability
   tests go red, and autosave comes back with evidence attached.

## Consequences

**Positive**

- The safety property is real and tested, instead of being a service that appears to provide it.
- A crash loses only the operation in flight, which SQLite's transaction rolls back — the state on
  disk is always a state the application produced.
- One scheduled task is closed by understanding rather than code, with the evidence in the suite.
- A future regression toward buffering is a red test rather than a design discussion.

**Negative**

- Every edit is a disk write, so a hypothetical editor issuing an update per mouse-move would
  produce a write per frame. That is a reason for M6 to issue one command per gesture (ADR-0045
  §6 already says so), not a reason to buffer.
- "No autosave" reads as a missing feature to anyone who does not read this ADR, which is why
  `Architecture.md` §4.5 is corrected in the same commit rather than left promising a service.

**Neutral**

- The `SIGKILL` test spawns a process, so it carries the `slow` marker and stays out of the inner
  loop.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A timer that calls `commit()` | There is no open transaction between calls to commit |
| A dirty-state buffer plus a flusher | Inventing the problem in order to solve it — batching writes and then autosaving them is a net loss of safety |
| An in-memory working copy saved periodically | The architecture ADR-0003 rejected: a project would stop being readable by a file manager while the application has it open |
| Write the service anyway, as a no-op, "for the shape" | A component whose entire behaviour is a comment, and one that would be maintained forever |
| Say nothing and skip the task | The next reader finds "autosave" in the roadmap, no autosave in the code, and no way to tell which is wrong |

## Compliance

- `tests/integration/test_durability.py` is the executable form of this decision, including a
  process killed with `SIGKILL` mid-work.
- Any new repository write method must commit before returning; the per-path test is where that is
  noticed.
- `docs/Architecture.md` §4.5 states the write-through rule instead of promising a service.

## References

- `docs/Architecture.md` §4.5 — the sentence this decision corrects
- ADR-0043 (cancellation is asked for, not forced) — the transaction and locking work underneath
- ADR-0045 (undo is a session) — the in-memory state that stays in memory on purpose
- ADR-0003 (projects are directories) — why an in-memory working copy is not an option
