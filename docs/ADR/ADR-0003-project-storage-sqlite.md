# ADR-0003 — Projects are directories; SQLite stores metadata only

- **Status:** Accepted
- **Date:** 2026-08-03
- **Affects:** `infrastructure/persistence`, `infrastructure/storage` · M4

## Context

The application needs to hold, for each body of work: source scans, preprocessing
parameters, detections, segmentation masks, manual annotations, measurements, exports,
model weights, logs and settings — and reopen all of it later, in a form the operator
trusts.

Scientific data outlives the software that produced it. An operator must be able to
inspect, copy, back up, archive and share their work without the application, and
without a migration tool. Scan files are also large: `data/` holds 628 SPM files today.

The current code has no persistence of any kind. Notebook state is the storage layer.

## Decision

**A project is a plain directory** with a fixed layout:

```
MyProject/
├── images/         source scans, byte-identical to what the operator imported
├── annotations/    manual annotations (JSON) and painted masks (image files)
├── results/        detections, measurements, generated masks
├── exports/        generated CSV
├── models/         project-local model weights
├── logs/           run logs
├── cache/          derived artifacts — safe to delete at any time
└── database.sqlite metadata
```

**SQLite stores metadata only:** projects, image metadata, annotations index,
measurements, training history, logs, settings. Specifically:

- **No image binaries in the database.** Images stay files; the database stores relative
  paths, checksums and metadata.
- **No mask bitmaps in the database.** Same rule.
- All paths are **relative to the project root**, so the directory is movable.
- Every schema has a version and a forward migration. No destructive migrations.
- Anything under `cache/` can be deleted at any moment without data loss, and the
  application must reconstruct it on demand.

## Consequences

**Positive**

- The operator owns their data in a form they already understand. `rsync`, `zip`, a
  backup tool, or a file manager all work.
- Corruption is contained: a damaged `database.sqlite` costs metadata, not scans.
- SQLite gives transactions, queries and a single-file index with zero server, zero
  configuration and zero network — appropriate for a single-user desktop application.
- Large binaries never pass through the database, so it stays small and fast.
- Diffable, greppable project structure makes debugging and support far cheaper.

**Negative**

- Two sources of truth to keep consistent: the filesystem and the index. Deleting a file
  behind the application's back produces a dangling row; the repository layer must
  reconcile (a startup integrity check is required).
- Relative paths mean the application must be careful never to write an absolute path
  into the database.
- A project directory is not atomic: half-copied projects are possible.
- SQLite concurrency is limited — fine for one desktop user, not for future multi-writer
  scenarios (explicitly out of scope).

**Neutral**

- Project format becomes a public contract with a version number, specified in M4-T01.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Everything in SQLite, images as BLOBs | Multi-gigabyte opaque files. Loses filesystem tooling, complicates backup, and makes partial recovery impossible. Explicitly rejected by the project brief. |
| Single-file project (HDF5 / zip container) | Atomic and portable, but opaque; incremental writes and crash recovery are much harder, and every access needs the application. |
| JSON/YAML sidecar files, no database | Fine at 10 images, unusable at 10 000 measurements. No transactions, no queries, no referential integrity. |
| PostgreSQL or another server database | Requires installation and administration on a lab machine. Wrong for an offline single-user desktop tool. |
| Document store (TinyDB, LiteDB) | Same maintenance cost as SQLite without the transactional guarantees or the ubiquity. |

## Compliance

- No table stores a column of type BLOB holding image or mask data.
- Every stored path is relative; an integration test opens a project moved to a new
  directory and asserts everything resolves.
- Deleting `cache/` and reopening the project produces identical results.
- Schema version is checked on open; unknown newer versions are refused with a clear
  message rather than silently migrated.

## References

- `systempromt.md` (Project storage, Database)
- `docs/Architecture.md` §4.4
- `docs/TASKS.md` M4-T01, M4-T02, M4-T03
