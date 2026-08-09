# ADR-0038 — The project format is a versioned contract, with two version numbers

- **Status:** Accepted
- **Date:** 2026-08-09
- **Deciders:** operator + agent (M4-T01)
- **Affects:** `infrastructure/storage`, `core/errors` · M4 · every task that writes into a project

## Context

ADR-0003 decided that **a project is a plain directory** — images as files, SQLite for metadata
only, relative paths throughout, a `cache/` that may be deleted at any moment — and it deferred
one thing to this task, by name:

> *Project format becomes a public contract with a version number, specified in M4-T01.*

That deferral is now blocking. Everything else in M4 writes into a project: the schema (M4-T02),
the repositories (M4-T03), the lifecycle use cases (M4-T04/T05), CSV export (M4-T11), the log
sinks (M4-T14). If the layout is settled implicitly by whichever of them lands first, the format
becomes *whatever the code happens to do* — and unlike every other contract in this repository,
the operator's data sits on the far side of it. A wrong guess here is not a refactor; it is a
directory somebody cannot open in two years.

What is true today: no application layer exists (audit W1), so there is nothing to migrate and no
project directory in the wild. **This decision is free to make now and expensive to make later**,
which is precisely why it is the first task of the milestone.

## Decision

### 1. Two independent version numbers

- **`format_version`** describes the *directory*: what files and subdirectories exist and what
  they mean. It lives in the manifest.
- **`schema_version`** describes the *database tables*. It lives in the database, as SQLite's own
  `PRAGMA user_version`, and M4-T02 owns it.

They change for different reasons. Adding a column is not adding a directory, and a single number
covering both would make every schema bump falsely claim the layout had changed — which matters,
because the layout is what a *human being with a file manager* relies on.

The layout version must also be readable **without opening the database**, which one shared number
cannot deliver.

### 2. The manifest is the identity file

A directory is a project **if and only if** it contains `project.json` at its root. Never inferred
from the presence of `images/` or `database.sqlite`.

The manifest is JSON, indented, with three required fields — `name`, `format_version`,
`created_utc` — and it is written so that `cat project.json` is a complete answer to "what is
this?". ADR-0003's promise that the operator owns their data is only real if the identity of the
directory is legible without our software.

It is **not** in the database, and that follows from ADR-0003's own consequence — *"corruption is
contained: a damaged `database.sqlite` costs metadata, not scans"*. A project whose database is
unreadable must still be able to say what it is, or the containment is a slogan.

### 3. One integer, not a semantic version

`format_version` is an `int`, bumped only when a reader that does not know about the change would
**misread** a project. Additions a reader can ignore do not bump it.

Semver's major/minor distinction exists to describe compatibility to *many independent readers*.
There is one reader here, shipped with the writer, and it needs exactly one question answered:
*can I open this?* A three-part version would be three fields to compare and two of them dead.

### 4. The compatibility matrix

| The manifest says | The application does |
|---|---|
| the **same** version | opens |
| an **older** version | opens, and migrates forward when a migration exists (M4-T02 owns migrations; this ADR fixes only that old is not refused for being old) |
| a **newer** version | **refuses**, naming both versions and telling the operator to upgrade |
| no manifest / unparseable / a missing field / a non-integer version | **refuses** as "not a project directory", naming the path |

Newer is refused because *a forward migration cannot be written by the past*: opening a project
written by a later version means guessing what its fields mean, and a wrong guess writes the guess
back to disk. This is ADR-0003's compliance rule ("unknown newer versions are refused with a clear
message rather than silently migrated"), stated as a matrix and made executable.

A non-integer version is refused rather than coerced, and that is not pedantry: `"2"` compares as
neither newer nor older than `1`, and `True` **is** an `int` in Python. Both would sail past a
naive check as "compatible".

### 5. One error, and it is a `ValueError`

`ProjectFormatError` covers all four refusals, because they are one statement to the operator —
*this is not something I can open* — with the message saying which. It is a `ValueError` and not a
`FileNotFoundError` even when the manifest is absent: the claim is about the directory, which
exists. "There is no `project.json` here" is how a directory says it is not a project, not an
accident of a path.

It joins the ADR-0030 taxonomy rather than sitting beside it, so `except NanoscopeError` keeps
meaning "the library said no".

## Consequences

**Positive**

- The format is decided before anything writes it, which is the only moment it is free.
- A project identifies itself in one small text file, readable by a human, a script, or a backup
  tool — and readable when the database is not.
- Layout and schema evolve independently, so a migration touches the version it actually changed.
- Refusals name the path or both versions; the operator's next move is never "guess".
- The spec ships with the check that enforces it, so drift between `docs/ProjectFormat.md` and the
  code shows up as a red test rather than as a support question.

**Negative**

- Two versions are two things to bump, and getting one wrong is a real error mode. Mitigated by
  their being read at different moments by different code, which makes confusing them awkward.
- Refusing a newer project is a hard stop for an operator who downgraded the application — by
  design, but it will read as the software being unhelpful, and the message has to carry that
  weight.
- `project.json` duplicates a little of what the database will also know (the project's name).
  Accepted: the duplicate is the price of the manifest being readable on its own, and the manifest
  is authoritative for identity.

**Neutral**

- `FORMAT_VERSION` starts at 1 with no legacy to support. The first migration is hypothetical
  until M4-T02 writes one.

## Alternatives considered

| Alternative | Why not |
|---|---|
| One version covering directory and schema | Every schema bump would claim the directory changed, and the layout could not be read without opening the database |
| The version in the database only | A corrupt `database.sqlite` would leave a directory unable to say what it is — the opposite of ADR-0003's containment consequence |
| Semantic versioning (`1.2.0`) | Three fields to compare where one question is asked, by a single reader shipped with the writer |
| Infer "this is a project" from the layout (`images/` + `database.sqlite`) | Any directory could accidentally qualify, and the check would silently accept a half-copied project — ADR-0003 names non-atomic copies as a known hazard |
| Refuse older versions too, and require an explicit migration command | Honest, and hostile: the common case is the operator's own project from last month |
| Open a newer version read-only | Sounds generous, means guessing what unknown fields mean; and "read-only" is not enforceable once the cache is written |
| A `.nanoscope` marker file plus metadata in the database | Two files where one carries both identity and version, for no gain |

## Compliance

- `tests/unit/test_project_format.py` executes every row of the matrix, and asserts each refusal
  names the path or both versions.
- `docs/ProjectFormat.md` is the contract; `FORMAT_VERSION` in
  `nanoscope/infrastructure/storage/project_format.py` is the same number, and the round-trip test
  fails if the manifest stops being plain readable JSON.
- No code outside `infrastructure/storage` constructs a project path by string concatenation; the
  directory names are constants.
- A reviewer bumping `FORMAT_VERSION` must be able to name the reader that would misread a project
  without the bump. If they cannot, the change is an addition and does not bump it.

## References

- ADR-0003 (projects are directories; SQLite stores metadata only) — the layout and the deferral
- ADR-0030 (a typed error taxonomy at the entry) — where `ProjectFormatError` belongs
- `docs/ProjectFormat.md` — the contract this ADR decides the shape of
- `docs/Architecture.md` §4.4 · `docs/TASKS.md` M4-T01, M4-T02, M4-T03
