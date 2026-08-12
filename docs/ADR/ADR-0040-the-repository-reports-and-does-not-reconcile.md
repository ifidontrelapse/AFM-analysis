# ADR-0040 — The repository reports what it finds, and never reconciles by deleting

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T03)
- **Affects:** `core/entities`, `core/ports`, `infrastructure/storage` · M4 · every use case above it

## Context

ADR-0003 chose a project format with **two sources of truth** — the files on disk and the SQLite
index that describes them — and wrote down the price in the same document:

> *Two sources of truth to keep consistent: the filesystem and the index. Deleting a file behind
> the application's back produces a dangling row; the repository layer must reconcile (a startup
> integrity check is required).*

M4-T01 specified the directory, M4-T02 gave it tables, and neither could collect that debt: the
check needs a repository to live in. This task writes the repository, so the question "what does
the application do when the two disagree?" cannot be deferred again — and it is the kind of
question that answers itself badly by default, because the code that finds a dangling row is
holding a `DELETE` statement already.

## Decision

### 1. The integrity check reports, in both directions, and changes nothing

`check_integrity()` returns an `IntegrityReport`: rows whose file is absent, and files under
`images/` that no row claims. It deletes nothing and imports nothing.

**A missing file is not evidence of a deletion.** It is equally an unmounted drive, a half-finished
copy, a `mv` the operator is about to undo, or a network share that is slow to come back. The row,
meanwhile, carries the annotations and measurements the file does not — so the automatic response
that "cleans up" a dangling row destroys the more expensive half of the pair, on *open*, without
being asked. ADR-0003 forbids destructive migrations; silently destructive startup would be worse
than one, because it has no version number attached.

**An untracked file is not evidence of an import.** Adopting it would mean guessing that a file
someone dropped into `images/` was meant to be part of the project and inventing a modality for
it. ADR-0003 chose a layout an operator can manipulate with a file manager; the cost of that
choice is that we do not get to assume every file is ours.

Resolving a report is a use case with an operator behind it, not a side effect of opening.

### 2. Existence, not contents

The check compares whether files exist. It does not verify checksums, because that reads every
scan in the project on every open — `data/` here holds 628 SPM files, and a project is expected to
be larger.

The checksum stays in the row for the question that needs it (did this file change since it was
imported?), asked explicitly by whoever wants the answer. Stated here so the omission is a
decision rather than an oversight.

### 3. The repository computes the checksum; it is never a parameter

`add_image` takes a path and hashes the file it points at. A checksum accepted from a caller can
describe a different file, and then the only thing the row proves is that two callers agreed with
each other. One funnel, and `sha256_of` is public so a verifier computes it the same way — two
definitions of "the checksum" compare unequal for reasons that have nothing to do with the data.

The same argument makes `add_image` refuse a file that is not there: a row whose file does not
exist is precisely the dangling row §1 exists to report, and there is no reason to create one
deliberately.

### 4. Relative paths are enforced in the repository, with the `CHECK` as the backstop

The repository converts an absolute path inside the project to a relative one, and refuses a path
outside it or one that escapes upward with `..`. M4-T02's `CHECK (relative_path NOT LIKE '/%')`
stays.

Both, deliberately. The constraint is what makes the rule true of the *database* — including for a
writer that does not go through this class — and the repository is what makes the refusal a
sentence the operator can act on. An absolute path is accepted rather than rejected because the
caller usually has one; rejecting it would push the conversion into every caller, which is where
it goes wrong.

### 5. Queries return entities

`ImageRecord`, not `sqlite3.Row`. A Row is untyped, and returning one puts the database's
vocabulary into every layer above the adapter — including `application`, which is not allowed to
know that SQLite exists.

### 6. The `ProjectRepository` port arrives now, with its first adapter

`core/ports/__init__.py` committed to exactly this: *"the rest ship with their first adapter"*.
This is that moment, and the port is load-bearing rather than ceremonial — M4-T04's use cases live
in `application/`, which may import `core` and nothing else (Architecture §3.2), so typing a use
case against `SqliteProjectRepository` would put `infrastructure` on the application's import
list.

It is a `Protocol`: the adapter satisfies it without importing it, so the arrow still points
inward and mypy checks the shape structurally.

## Consequences

**Positive**

- The debt ADR-0003 recorded three months ago is collected, and executed by tests that delete a
  file behind the application's back and move the whole project.
- No startup path can destroy the operator's data, which is a property that is much easier to keep
  than to regain.
- A checksum in a row always describes the file the row points at.
- `application` can be written against a project without knowing SQLite exists.
- The relative-path rule now has one enforcement point with a readable message, plus a constraint
  under it.

**Negative**

- A project with dangling rows stays that way until something asks. A report nobody reads is a
  report that did nothing — so M4-T04's `OpenProject` and M5's project explorer are on the hook
  for surfacing it, and this ADR is where that obligation is written down.
- `check_integrity` is O(rows + files under `images/`) and walks the directory. Fine at the sizes
  in front of us; a project with 100 000 files would want it off the open path.
- Two lists in a report is a shape that will want a third entry (checksum mismatches) the moment
  someone asks for verification. Additive, and deliberately not built now.

**Neutral**

- `remove_image` leaves the file, so removing an image turns it into an untracked file. That is
  the honest consequence of §1 and not a bug: forgetting a scan and deleting it are different
  decisions.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Delete dangling rows automatically on open | Silent data loss triggered by an unmounted drive, taking the annotations and measurements with it |
| Import untracked files automatically | Guesses that a stray file belongs to the project, and invents its modality |
| Mark rows "missing" in the database instead of reporting | A write on open, to record something that may be untrue a second later, in the file most likely to be on the same unmounted volume |
| Verify checksums on open | Reads every scan in the project, every time, to answer a question nobody asked yet |
| Accept a checksum from the caller | A checksum that can describe a different file proves only that two callers agreed |
| Return `sqlite3.Row` and skip the entity | Untyped, and it puts the database's vocabulary in `application` |
| Defer the port to M4-T04 | The use case would be written against the concrete class first, and that import is the one Architecture §3.2 forbids |

## Compliance

- `tests/integration/test_project_repository.py` deletes a recorded file, asserts the report names
  it, and asserts **the row is still there afterwards**.
- The same file opens a project that has been **moved** and one that has been **copied**, and
  asserts every path resolves — ADR-0003's compliance clause, executed at last.
- Deleting `cache/` and reopening the project changes nothing, also from ADR-0003's compliance
  list.
- No stored path is absolute: the repository refuses one, and the schema's `CHECK` refuses it
  again.
- Nothing above `infrastructure/storage` imports `sqlite3`.

## References

- ADR-0003 (projects are directories; SQLite stores metadata only) — the debt this collects
- ADR-0038 (the project format is a versioned contract) · ADR-0039 (the schema and its migrations)
- ADR-0019 / ADR-0025 (an unknown scale is a state) — why `pixel_size_nm` round-trips as `None`
- `docs/Architecture.md` §3.2, §4.4 · `docs/ProjectFormat.md` §5 · `docs/TASKS.md` M4-T03, M4-T04
