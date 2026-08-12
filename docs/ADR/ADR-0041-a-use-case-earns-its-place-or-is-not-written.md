# ADR-0041 — A use case earns its place, or it is not written

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T04)
- **Affects:** `application/use_cases`, `core/ports`, `infrastructure/storage` · M4 · M5's GUI

## Context

M4-T04 was specified as five use cases: `CreateProject`, `OpenProject`, `CloseProject`,
`ImportImages`, `ListImages`. Three of them are one line each on top of the repository M4-T03
already wrote, and writing them anyway is the cheapest, most respectable way for an application
layer to become ceremony — a directory of files that forward calls, each of which has to be read,
tested, kept in sync, and eventually deleted by someone who is not sure whether anything depends
on it.

The milestone's first exit criterion is what actually has to be true:

> *A project can be created, opened, populated with images and closed — from Python, headless.*

That is a statement about capability, not about file count.

## Decision

### 1. Two use cases are written; three are not

| Named in the task | Written | Why |
|---|---|---|
| `OpenProject` | ✅ `open_project` | Opens *and reads the integrity report*, returning it with the images. ADR-0040 ended on the obligation that something must read that report; this is it |
| `ImportImages` | ✅ `import_images` | Real policy: a batch does not abort on a bad file, and the caller receives both halves of the outcome |
| `CreateProject` | ➖ `SqliteProjectRepository.create` | `mkdir`, a manifest and SQLite from end to end — `application` may import none of them (Architecture §3.2), and PROJECT_RULES §2.7 already says the composition root constructs adapters |
| `CloseProject` | ❌ | It would be `repo.close()` |
| `ListImages` | ❌ | It would be `repo.list_images()` |

A function that forwards one call to one object is not a layer; it is a second name for the same
method, and the second name is the thing that later disagrees with the first. Both come back the
moment closing or listing means more than closing or listing — autosave (M4-T09) is the likely
trigger for one of them.

This is the same judgement M2-T08 made when it specified seven ports and wrote one, and it is
recorded here for the same reason: so that "only two of the five exist" reads as a decision in six
months, not as an unfinished task.

### 2. The batch does not abort, and only our own errors are caught

`import_images` attempts every file. A failure is collected as an `ImportFailure` carrying the path
and the reason, and the caller gets `ImportReport(imported, failed)`.

An operator importing a folder of forty scans must not lose the thirty-nine that were fine because
the fortieth was a partial download. What to do about a partial success — retry, ignore, ask — is a
decision above this layer, and refusing to make it here is the point of returning both lists.

Only `NanoscopeError` is caught. A file this library rejects is *data*, and belongs in the report;
anything else is a bug in this application, and a bug that keeps going for another thirty-nine
files is a worse bug.

### 3. A name that is already taken is disambiguated, not refused

`scan.spm`, then `scan_1.spm`. Two different scans called `scan.spm` in two folders is the ordinary
shape of AFM work, so refusing the second is hostile and overwriting it is data loss. The check is
against the **filesystem**, not the index: an untracked file is still a file, and copying over one
would destroy data the project does not even claim to own.

`display_name` keeps the operator's name (`scan.spm`) rather than the disambiguated one — the
suffix is a filesystem detail, and the name they gave the file is what they will look for.

### 4. The same file imported twice becomes two images — deliberately

No deduplication by checksum. It needs a `UNIQUE` index, a migration, and an answer to *"are two
identical scans ever legitimate?"* — which is an operator's question, not an engineer's, and
guessing it wrong is a constraint on a table that already has rows in it.

What it must not be is silent, and it is not: two rows in a list are visible, and `remove_image`
undoes it. Recorded here so the absence is a decision with a trigger — the first operator who asks
for it, or M4-T07's annotations, which make a duplicated image expensive rather than merely untidy.

### 5. Creating refuses a directory that is not empty

A new directory, or an existing empty one. Anything else is refused, naming the path: writing a
manifest into a folder that has files in it turns somebody else's `Documents/` into a project
directory, and the format explicitly says a directory *is* a project if it has a manifest
(ADR-0038).

## Consequences

**Positive**

- The milestone's first exit criterion is executable and executed, end to end, headless.
- The application layer contains only code that decides something. Everything else is one call to
  the port.
- A partial import is a first-class outcome rather than an exception the caller has to reconstruct.
- Two files with the same name can both be imported, which is what the work actually looks like.

**Negative**

- The GUI (M5) will hold a `ProjectRepository` and call `close()` and `list_images()` on it
  directly, which reads as skipping a layer until you notice the layer would have been a rename.
  If M5 finds itself wrapping either call, that is the signal to write the use case.
- Duplicate imports are possible and cost disk. Accepted, with §4's trigger.
- `create` on the adapter means the composition root — not a use case — is what a caller reaches
  for first. That is what PROJECT_RULES §2.7 says, and it will still surprise someone.

**Neutral**

- `import_image` on the port exists because the copy is filesystem work. The port therefore
  describes a slightly bigger surface than "an index", which is honest: it is a project's images,
  and getting a file into the project is part of that.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Write all five use cases | Three of them are renames; the layer becomes files to keep in sync with nothing in them |
| `create_project` as a use case, with a factory port | A port whose only implementation is passed to a function that immediately calls it — an interface written to satisfy a diagram |
| `import_images` raises on the first bad file | Loses thirty-nine good scans to the fortieth, and makes the caller re-derive what succeeded |
| Catch every exception in the batch | A bug in our code would be reported as a bad file, forty times |
| Refuse an import whose name collides | Hostile, and the collision is the norm rather than the exception |
| Overwrite on collision | Silent data loss, in the one directory the operator is promised is theirs |
| Deduplicate by checksum now | A `UNIQUE` index and a migration, to enforce an answer only an operator can give |
| Let `create` write into a non-empty directory | Any folder with a manifest dropped in it becomes a project (ADR-0038 §2) |

## Compliance

- `tests/integration/test_project_lifecycle.py` runs the exit criterion: create, import a folder,
  close, reopen, list — and asserts the layout on disk is the one `ProjectFormat.md` §1 specifies.
- The same file proves a failed import leaves **nothing** behind: no copied file, no row, and a
  clean integrity report.
- `tests/unit/test_project_use_cases.py` drives both use cases against a **second implementation**
  of `ProjectRepository`, which is what stops the port from being a type alias for one class.
- A `TypeError` from a repository escapes `import_images`; a `NanoscopeError` becomes a row in the
  report. Both are tested.
- `application` imports no adapter, no `sqlite3` and no filesystem module.

## References

- ADR-0040 (the repository reports and does not reconcile) — the obligation `open_project` discharges
- ADR-0038 (the project format is a versioned contract) · ADR-0039 (the schema and its migrations)
- M2-T08's port table in `core/ports/__init__.py` — the same judgement, made once before
- `docs/Architecture.md` §3.2 · `docs/Roadmap.md` M4 · `docs/TASKS.md` M4-T04
