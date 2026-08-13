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
| [0007](ADR-0007-park-web-client.md) | The React web client is parked; the desktop app is the product | ~~Superseded by 0012~~ | 2026-08-03 |
| [0008](ADR-0008-characterization-as-contract.md) | The characterization golden file is the refactor contract | Accepted | 2026-08-03 |
| [0009](ADR-0009-repository-hygiene.md) | No build artifacts, weights, or datasets in git | Accepted | 2026-08-03 |
| [0010](ADR-0010-isolated-numerical-changes.md) | One defect, one commit, one ADR, one golden update | Accepted | 2026-08-03 |
| [0011](ADR-0011-package-name-and-layout.md) | Package name and import layout — `nanoscope` | Accepted | 2026-08-03 |
| [0012](ADR-0012-delete-the-parked-client-and-batch-script.md) | Delete the parked React client and the broken batch script | Accepted | 2026-08-04 |
| [0013](ADR-0013-stdlib-logging-instead-of-a-logsink-port.md) | Standard-library `logging` instead of a `LogSink` port | Accepted | 2026-08-04 |
| [0014](ADR-0014-manual-opening-radius-is-the-radius-used.md) | `build_substrate_map` reports the opening radius it was given | Accepted | 2026-08-04 |
| [0015](ADR-0015-yolo-input-is-normalised-before-it-is-cast.md) | YOLO input is normalised before it is cast to `uint8` | Accepted | 2026-08-05 |
| [0016](ADR-0016-letterbox-the-yolo-input.md) | The YOLO input is letterboxed, not squashed | Accepted | 2026-08-05 |
| [0017](ADR-0017-otsu-sizing-fails-loudly-and-counts-what-it-kept.md) | Otsu sizing fails loudly, and counts what it kept | Accepted | 2026-08-05 |
| [0018](ADR-0018-log-normalisation-requires-a-positive-maximum.md) | LoG normalisation requires a positive maximum | Accepted | 2026-08-05 |
| [0019](ADR-0019-unknown-pixel-scale-is-a-state-not-a-crash.md) | An unknown pixel scale is a state, not a crash | Accepted | 2026-08-05 |
| [0020](ADR-0020-opening-radii-are-integers-rounded-up.md) | Opening radii are integers, rounded up (B4) | Accepted | 2026-08-05 |
| [0021](ADR-0021-the-tiled-backend-is-not-the-default.md) | The tiled YOLO backend is not the default (B7) | Accepted | 2026-08-05 |
| [0022](ADR-0022-the-golden-compares-messages-we-wrote.md) | The golden compares the messages we wrote, and only those | Accepted | 2026-08-05 |
| [0023](ADR-0023-polarity-is-configured-with-a-per-modality-default.md) | Polarity is configured, with a per-modality default (B3) | Accepted | 2026-08-05 |
| [0024](ADR-0024-the-minimum-particle-size-is-a-physical-size.md) | The minimum particle size is a physical size (B2) | Accepted | 2026-08-06 |
| [0025](ADR-0025-an-unknown-afm-scale-is-not-a-fabricated-one.md) | An unknown AFM scale is not a fabricated 1.0 | Accepted | 2026-08-06 |
| [0026](ADR-0026-a-header-without-a-scan-size-parses.md) | A header without a scan size parses | Accepted | 2026-08-06 |
| [0027](ADR-0027-an-empty-measurement-table-keeps-its-columns.md) | An empty measurement table keeps its columns | Accepted | 2026-08-06 |
| [0028](ADR-0028-a-detection-carries-its-own-score.md) | A detection carries its own score, or none | Accepted | 2026-08-06 |
| [0029](ADR-0029-flatten-lines-promotes-like-flatten-plane.md) | `flatten_lines` promotes the way `flatten_plane` does | Accepted | 2026-08-07 |
| [0030](ADR-0030-a-typed-error-taxonomy-at-the-entry.md) | A typed error taxonomy, checked at the entry | Accepted | 2026-08-07 |
| [0031](ADR-0031-one-measurement-schema.md) | One measurement schema, and a `bbox` that means something | Accepted | 2026-08-07 |
| [0032](ADR-0032-scoring-a-detector-against-ground-truth.md) | Scoring a detector against ground truth | Accepted | 2026-08-07 |
| [0033](ADR-0033-a-nan-height-is-not-a-measurement.md) | A height that is not a number is not a measurement | Accepted | 2026-08-07 |
| [0034](ADR-0034-a-sub-pixel-rough-radius-is-not-an-estimate.md) | A rough radius below one pixel is not an estimate | Accepted | 2026-08-07 |
| [0035](ADR-0035-the-rough-estimate-does-not-round-itself.md) | The rough estimate does not round its own radius | Accepted | 2026-08-08 |
| [0036](ADR-0036-levelling-can-fit-around-a-gap.md) | Levelling can fit around a gap | Accepted | 2026-08-08 |
| [0037](ADR-0037-the-opening-radius-constants-are-measured.md) | The opening-radius constants are named, exposed and measured | Accepted | 2026-08-08 |
| [0038](ADR-0038-the-project-format-is-a-versioned-contract.md) | The project format is a versioned contract, with two version numbers | Accepted | 2026-08-09 |
| [0039](ADR-0039-the-schema-has-a-version-and-a-way-forward.md) | The schema has a version and a way forward, and only the tables it has readers for | Accepted | 2026-08-12 |
| [0040](ADR-0040-the-repository-reports-and-does-not-reconcile.md) | The repository reports what it finds, and never reconciles by deleting | Accepted | 2026-08-12 |
| [0041](ADR-0041-a-use-case-earns-its-place-or-is-not-written.md) | A use case earns its place, or it is not written | Accepted | 2026-08-12 |
| [0042](ADR-0042-the-index-is-in-the-database-the-table-is-a-file.md) | The index is in the database, the measurement table is a file | Accepted | 2026-08-12 |
| [0043](ADR-0043-cancellation-is-asked-for-not-forced.md) | Cancellation is asked for, not forced | Accepted | 2026-08-12 |
| [0044](ADR-0044-an-annotation-is-a-row-because-it-cannot-be-recomputed.md) | An annotation is a row, because it cannot be recomputed | Accepted | 2026-08-12 |
| [0045](ADR-0045-undo-is-a-session-and-an-annotation-keeps-its-id.md) | Undo is a session, and an annotation keeps its id | Accepted | 2026-08-12 |
| [0046](ADR-0046-write-through-storage-needs-no-autosave.md) | Write-through storage needs no autosave | Accepted | 2026-08-12 |
| [0047](ADR-0047-a-preference-belongs-to-the-operator-or-to-the-work.md) | A preference belongs either to the operator or to the work | Accepted | 2026-08-12 |
| [0048](ADR-0048-an-export-is-not-a-copy-of-the-stored-table.md) | An export is not a copy of the stored table | Accepted | 2026-08-12 |
| [0049](ADR-0049-no-torch-is-a-cpu-and-rocm-is-not-cuda.md) | No torch is a CPU, and ROCm is not CUDA | Accepted | 2026-08-12 |
| [0050](ADR-0050-a-model-is-a-record-and-the-registry-hands-back-factories.md) | A model is a record, and the registry hands back factories | Accepted | 2026-08-12 |
| [0051](ADR-0051-a-log-must-not-live-inside-what-it-reports-on.md) | A log must not live inside the thing whose failure it reports | Accepted | 2026-08-12 |
| [0052](ADR-0052-the-entry-point-works-before-there-is-a-window.md) | The entry point works before there is a window | Accepted | 2026-08-12 |
| [0053](ADR-0053-qt-starts-behind-the-launcher-and-the-layout-is-a-setting.md) | Qt starts behind the launcher, and the layout is a setting | Accepted | 2026-08-12 |
| [0054](ADR-0054-one-source-of-colour-truth-and-a-contrast-floor.md) | One source of colour truth, and a contrast floor that can fail | Accepted | 2026-08-13 |
| [0055](ADR-0055-a-confirmation-that-counts-what-it-would-cost.md) | A confirmation that counts what it would cost | Accepted | 2026-08-13 |
| [0056](ADR-0056-the-viewer-shows-the-file-and-says-what-it-cannot-know.md) | The viewer shows the file, and says what it cannot know | Accepted | 2026-08-13 |
| [0057](ADR-0057-one-viewmodel-and-a-widget-that-emits-intent.md) | One viewmodel, and a widget that emits intent | Accepted | 2026-08-13 |
| [0058](ADR-0058-a-queued-signal-is-the-whole-of-the-marshalling.md) | A queued signal is the whole of the marshalling | Accepted | 2026-08-13 |
| [0059](ADR-0059-a-log-line-is-a-fact-and-a-notification-is-a-count.md) | A log line is a fact, and a notification is a count | Accepted | 2026-08-13 |
| [0060](ADR-0060-a-setting-is-offered-only-if-something-reads-it.md) | A setting is offered only if something reads it | Accepted | 2026-08-13 |
| [0061](ADR-0061-a-preview-is-asked-for-and-names-the-array-it-shows.md) | A preview is asked for, and names the array it shows | Accepted | 2026-08-13 |
| [0062](ADR-0062-the-matrix-decides-what-may-be-asked-for.md) | The matrix decides what may be asked for | Accepted | 2026-08-13 |
| [0063](ADR-0063-the-overlay-draws-what-the-detection-carries.md) | The overlay draws what the detection carries | Accepted | 2026-08-13 |
| [0064](ADR-0064-the-predictor-is-built-once-and-the-masks-are-not-stored.md) | The predictor is built once, and the masks are not stored | Accepted | 2026-08-13 |
| [0065](ADR-0065-a-row-and-a-particle-are-linked-by-position.md) | A row and a particle are linked by position | Accepted | 2026-08-13 |
| [0066](ADR-0066-statistics-are-computed-below-the-widget.md) | Statistics are computed below the widget | Accepted | 2026-08-13 |
| [0067](ADR-0067-an-export-names-its-scope-and-lands-in-the-project.md) | An export names its scope, and lands in the project | Accepted | 2026-08-13 |
| [0068](ADR-0068-navigation-is-the-same-selection-and-it-does-not-wrap.md) | Navigation is the same selection, and it does not wrap | Accepted | 2026-08-13 |
| [0069](ADR-0069-what-persists-is-the-run-not-the-session.md) | What persists is the run, not the session | Accepted | 2026-08-13 |
| [0070](ADR-0070-hand-work-is-drawn-as-hand-work.md) | Hand work is drawn as hand work | Accepted | 2026-08-13 |
| [0071](ADR-0071-the-shape-is-a-box-and-a-tool-does-not-invent-one.md) | The shape is a box, and a tool does not invent one | Accepted | 2026-08-13 |
| [0072](ADR-0072-an-outline-is-stored-beside-the-box.md) | An outline is stored beside the box | Accepted | 2026-08-14 |
| [0073](ADR-0073-a-painted-mask-is-a-file-and-the-row-points-at-it.md) | A painted mask is a file, and the row points at it | Accepted | 2026-08-14 |
| [0074](ADR-0074-a-ruler-is-not-an-annotation.md) | A ruler is not an annotation, and the word "measurement" is taken | Accepted | 2026-08-14 |
| [0075](ADR-0075-a-profile-is-the-notebooks-slice-extended-with-a-stated-rule.md) | A profile is the notebook's slice, extended with a stated rule | Accepted | 2026-08-14 |
| [0076](ADR-0076-a-detection-is-a-record-and-an-annotation-is-the-edit.md) | A detection is a record; an annotation is the edit | Accepted | 2026-08-14 |

## When to write one

Write an ADR when you had to choose between two viable ways of doing something and the
choice will be expensive to reverse. Examples: a layer boundary, a storage format, a
dependency, a threading model, a numerical behaviour change.

Do not write one for: a naming preference, a refactor with no external contract change,
or anything a test already documents better.
