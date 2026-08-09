# CURRENT TASK

**ID:** `M4-T01`
**Title:** The project directory format as a versioned public contract
**Milestone:** M4 — Application layer, first task
**Defect:** — (W1: no application layer exists) · **ADR:** **ADR-0038** (to be written)
**Branch:** `feat/m4-application-layer` — M4 changes no scientific output, so the `sci/` prefix
no longer applies (PROJECT_RULES §7)
**Status:** **planned 2026-08-09.**

---

## Why this task is first

Everything else in M4 writes into a project: the SQLite schema (M4-T02), the repositories
(M4-T03), the use cases (M4-T04/T05), exports (M4-T11), logs (M4-T14). If the layout is decided
implicitly by whichever of them lands first, the format ends up as "whatever the code does" — and
ADR-0003 already committed the opposite in writing:

> *Project format becomes a public contract with a version number, specified in M4-T01.*

ADR-0003 fixed the **layout** (a directory, images as files, SQLite for metadata only, relative
paths, a disposable `cache/`) and deliberately deferred the **contract**: what the version number
means, what happens when the application meets a project it does not recognise, and what a reader
is allowed to assume about a directory that claims to be a project.

**The risk profile of M4 is the inverse of M3's.** The scientific core is called, not modified,
so no task here should move a golden number at all — a red golden in M4 is a bug in M4.

---

## The decisions this task has to make

**1. One version or two?** The directory *layout* and the SQLite *schema* can change
independently — adding an `exports/` convention is not adding a column. ADR-0003 says "every
schema has a version" and says nothing about the layout's.

| | |
|---|---|
| **Two independent versions** ✅ | `format_version` in the manifest describes the directory; `schema_version` in the database describes the tables (`PRAGMA user_version`, which SQLite already provides). They change for different reasons and are read at different moments — the layout must be readable *without opening the database*, which is the whole point of a directory the operator owns |
| One version covering both | Simpler to state, and wrong the first time a migration touches only one of them: every schema bump would falsely claim the directory changed |

**2. Where does the format version live?** In a `project.json` manifest at the project root —
**not** in the database. A project whose `database.sqlite` is corrupt must still identify itself,
which is ADR-0003's own "corruption is contained" consequence taken seriously.

**3. What does an unrecognised version do?** ADR-0003's compliance section already states the
rule for newer versions ("refused with a clear message rather than silently migrated"). This task
states the full matrix and makes it executable:

| The project says | The application does |
|---|---|
| a **newer** major version | refuse, naming both versions — a forward migration cannot be written by the past |
| an **older** version | open, and migrate forward when a migration exists (M4-T02 owns migrations; this task owns the *rule*) |
| the **same** version | open |
| no manifest, or unparseable | refuse as "not a project directory" — never guess from the presence of `images/` |

---

## Scope

**In scope**

1. `docs/ProjectFormat.md` — the contract: layout, the manifest's fields and their meanings,
   the two version numbers, the compatibility matrix, the path rules (relative, always), what is
   authoritative when the filesystem and the index disagree, and what `cache/` guarantees
2. **ADR-0038** — two independent versions, the manifest as the identity file, refuse-newer
3. `nanoscope/infrastructure/storage/project_format.py` — the executable half:
   `FORMAT_VERSION`, the directory names as constants, a `ProjectManifest` dataclass,
   `read_manifest`, `write_manifest`, and one `check_compatible` that implements the matrix and
   raises a typed error naming both versions
4. `ProjectFormatError` in `core/errors.py` — Architecture §4.6 already lists `StorageError` in
   the target taxonomy; this is that slot, named for what it actually reports
5. Tests over the matrix: same / older / newer / absent / unparseable, and a round trip

**Out of scope**

- **Creating a project.** `CreateProject` is M4-T04 and needs the repository underneath it. This
  task can describe and validate a project directory; it does not own the lifecycle
- **The SQLite schema and its migrations** — M4-T02. This task fixes only that a schema version
  exists, where it lives, and what an unknown one does
- **An integrity check that reconciles dangling rows** — ADR-0003 requires one; it belongs with
  the repository in M4-T03, because it needs the tables to check against

---

## Why the spec ships with code

A specification nothing executes drifts from the code within two tasks, and this one is a
*contract* — the case where drift is most expensive, because the operator's data is on the other
side of it. The code here is deliberately thin: constants, a manifest, and the version check.
Everything that needs a database waits for the task that has one.

---

## Expected blast radius

- **Zero golden differences**, and no numerical code is touched at all. If the golden moves,
  something imported the science by accident
- One new module, one new error class, one new document, one ADR

---

## Definition of done

- [ ] `docs/ProjectFormat.md` — the contract, versioned, with the compatibility matrix
- [ ] ADR-0038 recording the three decisions and what was rejected
- [ ] `project_format.py` — constants, manifest read/write, `check_compatible`
- [ ] `ProjectFormatError`, in the existing taxonomy rather than beside it
- [ ] Tests over the whole matrix, including the unparseable manifest
- [ ] `make check` green; golden byte-identical
- [ ] `STATE.md`, `Progress.md`, `TASKS.md`, `Architecture.md` §4.4 (pointing at the spec),
      `PROJECT_CONTEXT.md`, ADR index
- [ ] Commit: `M4-T01: the project format is a versioned contract`
