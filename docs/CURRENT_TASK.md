# CURRENT TASK

**ID:** `M4-T04`
**Title:** The project lifecycle: create, open, import — and the two use cases not worth writing
**Milestone:** M4 — Application layer, fourth task
**Defect:** — (W1: no application layer exists) · **ADR:** **ADR-0041** (to be written)
**Branch:** `feat/m4-application-layer` — M4 changes no scientific output (PROJECT_RULES §7)
**Status:** planned 2026-08-12, implementation next.

---

## Why this task is next

M4-T03 left two things on the table by name — creating the directory, and copying an imported file
into `images/` before it is recorded — and the milestone's first exit criterion is exactly the
sentence they complete:

> *A project can be created, opened, populated with images and closed — from Python, headless.*

Everything needed is now underneath it: the format (T01), the tables (T02), the index and the
integrity check (T03). This is the task where they become something an operator can run.

---

## The decisions this task has to make

**1. Where does "create a project" live?** In `infrastructure`, as
`SqliteProjectRepository.create`.

Creating a project is `mkdir`, a manifest and a database — filesystem and SQLite from end to end,
and Architecture §3.2 says `application` may import neither. Putting it in a use case would mean
either breaking that rule or inventing a factory port whose only job is to be passed to a function
that immediately calls it. PROJECT_RULES §2.7 already says who constructs adapters: the
composition root.

**2. Then what is left for `application`?** Two things with actual policy in them, and this is the
task's real question — a use case that forwards one call to one object is not a layer, it is a
second name for the same method.

| Use case | Verdict |
|---|---|
| **`open_project`** ✅ | Opens *and reads the integrity report*, returning the project's images with it. ADR-0040 ended with an obligation — "a report nobody reads is a report that did nothing" — and this is where it is discharged |
| **`import_images`** ✅ | Real policy: a batch of forty scans must not abort on the third one, so each file succeeds or fails on its own and the caller gets both lists |
| `close_project` ❌ | `repo.close()`. A wrapper would add a name and nothing else. It comes back the moment closing means something more than closing — autosave is M4-T09 |
| `list_images` ❌ | `repo.list_images()`. Same |

Writing the two that are empty is how a layer becomes ceremony, and this repository has already
made that call once, deliberately, in M2-T08: *seven ports were specified, one was written.*

**3. What does the copy do about a name that is already taken?** Disambiguates —
`scan.spm`, `scan_1.spm`. Two different scans called `scan.spm` from two folders is the normal
shape of AFM work, and refusing the second is hostile.

**4. And about the *same file* imported twice?** Nothing, on purpose. It becomes a second copy
with a second row, visible as two rows in a list rather than as silent corruption.

Deduplicating by checksum wants a `UNIQUE` index, a migration, and an answer to "are two identical
scans ever legitimate?" — which is an operator's question, not an engineer's. Deferred out loud in
the ADR rather than guessed at now.

**5. What happens when `create` is pointed at a directory that already has something in it?**
Refused. An empty directory or one that does not exist yet is fine; anything else is refused
naming the path, because writing a manifest into somebody's `Documents/` folder makes it a
project directory.

**6. Does the failure of one file abort the import?** No, and that is the whole reason
`import_images` exists. Each failure is collected with the reason and the path, the rest of the
batch continues, and the caller decides what to do with a partial success.

---

## Scope

**In scope**

1. `SqliteProjectRepository.create(directory, name)` — the layout, the manifest, the database
2. `import_image(source, …)` on the repository and the port — copy into `images/` under a free
   name, then record. Copying is infrastructure's, and it must happen before the row exists
3. `application/use_cases/projects.py` — `open_project`, `import_images`
4. `OpenedProject`, `ImportReport`, `ImportFailure` in `core/entities/project.py`
5. **ADR-0041** — what earns a use case, where `create` lives, the batch that does not abort, the
   deduplication deferred
6. Tests: the use cases against a **fake repository** (which is also the proof that the port is
   usable by something other than SQLite), and an integration test that runs the exit criterion
   end to end — create, import, close, reopen, list

**Out of scope**

- **Jobs and progress** — M4-T06. `import_images` returns when it is done; a forty-file import
  that needs a progress bar is that task's problem
- **Undo** — M4-T08, which is what `remove_image` is waiting for
- **Deduplication by content** — decision 4, deferred with its reason
- **A composition root beyond one line** — `app/` gets its wiring in M5, with an entry point to
  wire

---

## Expected blast radius

- **Zero golden differences.** No numerical code is imported
- One new application module, one new entity trio, one ADR, two test files
- No new dependency — `shutil` is stdlib

---

## Definition of done

- [ ] `create` and `import_image`, with the port extended to match
- [ ] `open_project` and `import_images`, with `close_project` and `list_images` deliberately absent
- [ ] ADR-0041, including what was *not* written and why
- [ ] Use-case tests against a fake repository; an end-to-end test of the exit criterion
- [ ] `make check` green, golden byte-identical
- [ ] `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, `Roadmap.md` (the criterion),
      ADR index
- [ ] Commit: `M4-T04: a project can be created, opened and populated`
