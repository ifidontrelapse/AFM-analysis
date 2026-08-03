# Architecture Decision Records

One file per decision. Numbered sequentially, never renumbered, never reused.
An accepted ADR is immutable — to change a decision, write a new ADR and mark the old
one `Superseded by ADR-XXXX`.

Use `TEMPLATE.md` for new records.

| # | Title | Status | Date |
|---|---|---|---|
| [0001](ADR-0001-clean-architecture.md) | Clean Architecture with a preserved scientific domain | Accepted | 2026-08-03 |
| [0002](ADR-0002-qt6-desktop-ui.md) | Qt6 / PySide6 desktop application, dark theme only | Accepted | 2026-08-03 |
| [0003](ADR-0003-project-storage-sqlite.md) | Projects are directories; SQLite stores metadata only | Accepted | 2026-08-03 |
| [0004](ADR-0004-device-manager.md) | A single Device Manager owns backend selection | Accepted | 2026-08-03 |
| [0005](ADR-0005-pluggable-model-providers.md) | Models are pluggable providers behind ports and a registry | Accepted | 2026-08-03 |
| [0006](ADR-0006-training-provider.md) | Training is an application module behind `TrainingProvider` | Accepted | 2026-08-03 |
| [0007](ADR-0007-park-web-client.md) | The React web client is parked; the desktop app is the product | Accepted | 2026-08-03 |
| [0008](ADR-0008-characterization-as-contract.md) | The characterization golden file is the refactor contract | Accepted | 2026-08-03 |
| [0009](ADR-0009-repository-hygiene.md) | No build artifacts, weights, or datasets in git | Accepted | 2026-08-03 |
| [0010](ADR-0010-isolated-numerical-changes.md) | One defect, one commit, one ADR, one golden update | Accepted | 2026-08-03 |
| [0011](ADR-0011-package-name-and-layout.md) | Package name and import layout | **Proposed** | 2026-08-03 |

## When to write one

Write an ADR when you had to choose between two viable ways of doing something and the
choice will be expensive to reverse. Examples: a layer boundary, a storage format, a
dependency, a threading model, a numerical behaviour change.

Do not write one for: a naming preference, a refactor with no external contract change,
or anything a test already documents better.
