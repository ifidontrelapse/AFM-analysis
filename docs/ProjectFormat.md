# The nanoscope project format

**Format version:** 1 · **Status:** active · **Specified by:** M4-T01 · **Decided in:** ADR-0038,
on top of ADR-0003

This is a **public contract**. A project directory outlives the version of the application that
made it, and an operator must be able to inspect, copy, back up and archive their work without
us. Everything below is what a reader may assume and what a writer must produce.

The executable half of this document is
`nanoscope/infrastructure/storage/project_format.py`; every rule with a MUST in it is covered by
`tests/unit/test_project_format.py`.

---

## 1. Layout

```
MyProject/
├── project.json      the manifest — this file is what makes the directory a project
├── database.sqlite   metadata index; no image or mask binaries, ever
├── images/           source scans, byte-identical to what the operator imported
├── annotations/      manual annotations (JSON) and painted masks (image files)
├── results/          detections, measurements, generated masks
├── exports/          generated CSV
├── models/           project-local model weights
├── logs/             run logs
└── cache/            derived artifacts — safe to delete at any moment
```

- A directory **is a project if and only if** it contains `project.json` at its root. The presence
  of `images/` or `database.sqlite` means nothing on its own.
- Every path stored anywhere — in the manifest, in the database, in an annotation file — **MUST be
  relative to the project root**. The directory is movable, and an absolute path breaks that.
- `cache/` MAY be deleted at any time, by anyone, with no data loss. Reopening the project
  reconstructs whatever it needs. Nothing may be stored only in `cache/`.
- The source scans under `images/` are never modified in place.
- `database.sqlite` is the **only** database file. There is no `-wal` or `-shm` beside it: the
  application does not use WAL, so a copy of the directory is a complete copy of the index
  (ADR-0039 §4). A reader may rely on that when archiving a project.

## 2. The manifest

`project.json`, UTF-8, indented JSON, one object:

```json
{
  "name": "MyProject",
  "format_version": 1,
  "created_utc": "2026-08-09T12:00:00+00:00"
}
```

| Field | Type | Meaning |
|---|---|---|
| `name` | string | The project's display name. Independent of the directory name, which the operator may rename freely |
| `format_version` | integer | The version of *this document* that the directory conforms to |
| `created_utc` | string | ISO-8601, UTC, when the project was created. Informational |

Rules:

- All three fields are **required**. A reader MUST refuse a manifest missing any of them.
- `format_version` MUST be an integer. A string `"1"` or a boolean is refused, not coerced — see
  ADR-0038 §3 for why that is not pedantry.
- A reader MUST ignore fields it does not know, and MUST NOT delete them when it rewrites the
  manifest. That is what makes an additive change additive.
- The manifest is **authoritative for the project's identity**. Where it and the database
  disagree about the name, the manifest wins.

The manifest deliberately carries almost nothing. It has one job — *say what this directory is,
without opening the database* — and every field added to it is a field that must stay readable
forever.

## 3. Two version numbers

| Version | Lives in | Describes | Owned by |
|---|---|---|---|
| `format_version` | `project.json` | the directory: which files exist and what they mean | this document |
| `schema_version` | `database.sqlite`, as `PRAGMA user_version` | the database tables | `infrastructure/storage/database.py` (M4-T02, ADR-0039) |

They are independent and are bumped separately. Adding a column is not adding a directory.

`schema_version` is `0` for a database with no tables — an absent `database.sqlite` and a freshly
created one are the same state — and it advances one step at a time as the migrations run. A
newer `schema_version` is refused for the same reason a newer `format_version` is, with the same
error.

**`format_version` is bumped only when a reader that does not know about the change would
*misread* a project.** Adding an optional field, or a new subdirectory that older readers can
ignore, is not such a change and does not bump it.

## 4. Compatibility

| The manifest says | The application does |
|---|---|
| the same version | opens |
| an older version | opens, migrating forward if a migration is needed |
| a newer version | **refuses**, naming both versions, and says to upgrade the application |
| no manifest, unparseable JSON, not an object, a missing field, a non-integer version | **refuses** as "not a project directory", naming the path |

The same four rows hold for `schema_version`, read from the database instead of the manifest, and
a refusal is the same `ProjectFormatError`.

Migrations are **forward-only and never destructive**: a migration may add, and may rewrite what
it fully understands, but it may not discard data it does not recognise. A database migration is
one step at a time and **all-or-nothing** — the tables and the version number move together, so a
migration that fails leaves the database exactly as it was (ADR-0039 §2).

Refusing a newer project is deliberate. A forward migration cannot be written by the past —
opening a project written by a later version means guessing what its fields mean, and the guess
would be written back to disk.

## 5. What is authoritative

| Question | Answer |
|---|---|
| What is this directory? | `project.json` |
| Which images are in the project? | the database, reconciled against `images/` on open (M4-T03) |
| What does an image contain? | the file. The database stores its relative path, a checksum and metadata — never the pixels |
| What are the measurements? | the database |
| Anything under `cache/` | nothing. It is derived, and it is disposable |

The filesystem and the index are two sources of truth, and ADR-0003 names the cost: a file
deleted behind the application's back leaves a dangling row. The repository's integrity check
(M4-T03, ADR-0040) is where they are compared, and it **reports in both directions and changes
nothing** — a dangling row is not deleted, because a missing file is as likely to be an
unmounted drive as a deletion and the row carries measurements the file does not; a file no row
claims is not imported, because nothing here knows it was meant to be in the project. Resolving
a report is a decision with an operator behind it.

## 6. Errors

Every refusal in this document is a `ProjectFormatError` — one error class for four cases,
because to the operator they are one statement: *this is not something I can open*. The message
says which case it was and names the path or the two versions.

`ProjectFormatError` is a `NanoscopeError` and a `ValueError` (ADR-0030's taxonomy). It is not a
`FileNotFoundError` even when the manifest is absent: the claim is about the directory, which
exists.

## 7. Changing this document

1. Decide whether the change makes an existing reader **misread** a project. If it does, bump
   `format_version` here and in `project_format.py`, and write the migration in the same change.
   If it does not, it is an addition; say so explicitly in the commit.
2. Update the matrix in §4 if the behaviour changed, and the tests that execute it.
3. A new ADR is needed when the change reverses a decision in ADR-0003 or ADR-0038. Extending the
   layout does not need one.

Changing the **database** is a different move: append a step to `MIGRATIONS` in
`database.py`, never edit one that has shipped, and leave `format_version` alone — a new table is
not a new directory. `SCHEMA_VERSION` follows the list on its own.

## References

- **ADR-0003** — projects are directories; SQLite stores metadata only
- **ADR-0038** — the project format is a versioned contract, with two version numbers
- **ADR-0039** — the schema's version, its migrations, and why the directory holds one database file
- `docs/Architecture.md` §4.4 · `docs/TASKS.md` M4-T01…M4-T03
