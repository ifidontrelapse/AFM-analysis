# Backlog

Everything that is not on the roadmap yet. Nothing is ever thrown away — it is recorded
here with a reason and a trigger for revisiting.

**Statuses:** `idea` · `candidate` (likely to be scheduled) · `parked` (deliberately not
now) · `rejected` (with a reason) · `promoted → M#-T##`.

---

## Application features

| ID | Item | Status | Notes |
|---|---|---|---|
| B-001 | **Batch processing** — run a pipeline over a directory of scans | parked | Explicitly out of scope for v1 (`systempromt.md`). The job abstraction in M4-T06 is designed so this is additive, not a redesign. Revisit after M6. |
| B-002 | **Plugin system** — third-party detectors/segmenters/exporters | parked | Not required now. The port + registry layout (ADR-0005) makes it addable without touching the core. Trigger: a second external contributor, or a model we cannot vendor. |
| B-003 | Export formats beyond CSV — JSON, HDF5, Origin, XLSX | idea | CSV only for v1. HDF5 becomes interesting once masks are exported. |
| B-004 | Report generation — PDF/HTML with figures, statistics and provenance | idea | The natural companion to CSV export for publication workflows. |
| B-005 | Multi-image comparison view — same statistics across scans in a project | candidate | Requested implicitly by the "Statistics" module; small once M6 exists. |
| B-006 | Particle classification — size/shape classes, clustering | idea | Needs a scientific definition first. |
| B-007 | Scan stitching / large-area mosaics | idea | Hardware-dependent; low demand so far. |
| B-008 | Region of interest — restrict analysis to a drawn region | candidate | Cheap once the annotation layer (M7-T01) exists. |
| B-009 | Session restore — reopen the last project and layout on launch | candidate | Depends on M5-T02 layout persistence. |
| B-010 | Undo/redo history panel | idea | The command stack (M4-T08) already carries the data. |
| B-011 | Keyboard-first workflow + a command palette (VSCode-inspired) | candidate | Aligns with the stated design inspiration. |
| B-012 | Scriptable console inside the app (Python REPL against the open project) | idea | napari-inspired; powerful for the operator, but a security and support surface. |

---

## Scientific / algorithmic

| ID | Item | Status | Notes |
|---|---|---|---|
| B-020 | **Additional AFM formats** — `.ibw` (Igor), `.gwy` (Gwyddion), JPK, Park | candidate | The `load_afm` docstring already advertises `.ibw` and `.gwy`; they are not implemented. Fix the docstring in M3, implement here. |
| B-021 | Alternative flattening — higher-order polynomial, median-of-differences, facet levelling | idea | Gwyddion offers these; operators expect them. |
| B-022 | Substrate estimation alternatives — rolling ball, RANSAC plane, wavelet | idea | Morphological opening is the only method today, and it assumes the radius exceeds every particle. |
| B-023 | Watershed / distance-transform separation of touching particles | candidate | The `afm_dense_overlapping` phantom finds 59 of 70; this is where the loss is. |
| B-024 | Sub-pixel centroid refinement | idea | Improves diameter statistics on small particles. |
| B-025 | Volume and surface-area measurement | candidate | Natural extension of the height measurement; already possible from masks. |
| B-026 | Tip-convolution deconvolution | idea | The classic AFM systematic error — particles measure wider than they are. Scientifically valuable, needs tip geometry. |
| B-027 | Uncertainty estimation on every measurement | idea | Turns "23.05 nm" into "23.05 ± 0.4 nm". Publication-relevant. |
| B-028 | Drift/noise diagnostics per scan line | idea | Would explain measurement outliers instead of hiding them. |
| B-029 | SAM2 automatic mask generation (no prompts) | idea | Alternative to detector-prompted segmentation. |
| B-030 | Model ensembling / cross-detector agreement scores | idea | Depends on M3-T15 evaluation harness. |

---

## Engineering

| ID | Item | Status | Notes |
|---|---|---|---|
| B-040 | **Purge `node_modules` and weights from git history** | candidate | M1-T01 stops the bleeding; history still carries 78 MB. A rewrite affects the remote and needs operator approval. Trigger: clone time becomes a complaint. |
| B-041 | **Delete `frontend/`** | **done → ADR-0012** | Operator authorised deletion 2026-08-04. 21 tracked files removed; recoverable from history. |
| B-042 | **Delete or port `preprocess_batch.py`** | **done → ADR-0012** | Deleted 2026-08-04. Broken on every file since `e8caf25` (D-02) and nobody noticed, which settled the question. Batch processing stays out of scope (B-001); when it returns it is an entry point over the application layer, not a script importing `src`. |
| B-043 | Notebook policy — keep as documentation, or move to a separate repository | idea | 8.7 MB of committed notebooks with outputs. M1-T09 strips outputs; the policy question remains. |
| B-044 | Property-based tests for the numerical core (Hypothesis) | candidate | Degenerate inputs are exactly what property testing is for. After M3-T13. |
| B-045 | Benchmark suite + performance regression tracking | idea | LoG on a 2048² scan is the practical limit today; nobody has measured it. |
| B-046 | Type-checked frontend contract, if the web client is ever revived | rejected | The client is deleted (ADR-0012). Reviving it means a new ADR and starting from `docs/archive/plan-frontend-react-client.md`. |
| B-047 | Translations / i18n (Russian UI) | candidate | PROJECT_RULES requires English in source; user-facing strings go through a catalog precisely so this stays possible. |
| B-048 | Accessibility pass on the Qt UI | idea | Contrast, focus order, keyboard navigation. |
| B-049 | Telemetry-free crash reports the user can inspect before sending | candidate | M9-T04 builds the bundle; sending it is a separate decision. |
| B-050 | Windows / macOS support | parked | The target is Linux desktop. Qt6 and the domain layer are portable; `DeviceManager` already anticipates MPS. |
| B-051 | Reproducibility manifest — record library versions with every result | candidate | The golden file already does this for tests (`_meta`); results deserve the same. |
| B-052 | Commit a small real SPM scan as a test fixture | idea | Blocked on B6. The phantom set stands in for it today. |
| B-053 | Nanoscope parser hardening against other firmware versions | candidate | 120 of 628 local files parsed; the regexes are unverified elsewhere. Scheduled as M3-T16. |
| B-057 | Install `pandas-stubs` / `scipy-stubs` instead of silencing those imports | candidate | M1-T04 scoped `ignore_missing_imports` to pandas and scipy rather than adding stub packages: real stubs against pandas 2.x typically surface a fresh wave of errors, and `src/` is deleted in M2-T15. Revisit once `nanoscope` exists and is strict — there the stubs pay for themselves. |
| B-056 | Consider the `S` (bandit) ruff rule family | idea | `per-file-ignores` carried `"tests/*" = [..., "S101"]` while `S` was never selected — dead configuration, removed in M1-T03. Selecting `S` would add security lint; it also changes the M2 burn-down baseline, so it is a decision for after M2, not during. |
| B-055 | Declare `clip` if YOLO-World support is wanted | candidate | `clip`, `ftfy` and `regex` were installed outside uv and were removed by `uv sync` in M1-T02. Nothing in `src/` imports them; they are needed only by YOLO-World models such as `checkpoints/yolov8s-world.pt`, which is not the configured default. Either declare `clip @ git+https://github.com/ultralytics/CLIP.git` as a dependency or drop YOLO-World. Operator's call. |
| B-058 | **The golden records CPython exception messages verbatim, so it is pinned to the interpreter's minor version** | **done → ADR-0022** (2026-08-05) | The type and the raising function are still compared exactly; the message is compared only when `_we_wrote_this_message` says this project typed it — the frame must be inside `nanoscope` **and** the raising line must be an explicit `raise`, because either signal alone misclassifies (`h, w = z.shape` in our file is CPython's wording; skimage also raises explicitly). 15 keys renamed to `error_message_unchecked`, which `compare` skips; 7 remain compared, all `estimate_radius_otsu`'s. **A Python upgrade no longer reads as drift.** |
| B-059 | **A NaN height passes the non-positive filter and reaches the measurement table** | candidate — needs its own task | `measure_all_baseline` drops a particle with `if metrics["height_nm"] <= 0`, and `nan <= 0` is `False`, so a NaN row survives. Reachable on a constant map: `substrate_mask` is empty, `np.median` of nothing is `nan`, and `global_baseline` carries it into every height. **ADR-0018 already ruled on this exact comparison** — the guard must be `not height > 0`, because that and `<= 0` differ precisely on `nan`. Found while writing M3-T12's tests; not fixed there, because it moves a number and ADR-0010 keeps one defect to one commit. |
| B-060 | **Levelling that fits around a dropped scan line instead of refusing it** | candidate — needs its own task | M3-T13 (ADR-0030) made a non-finite value a rejection: a height map must be finite, because `flatten_plane` already enforced that through `scipy.lstsq` while `flatten_lines` propagated the NaN and `detect_particles` answered "no particles". Rejecting is the honest reading of what the code did; it is not the best one. A dropped scan line is a real artefact, and a masked least-squares fit — `lstsq` on the finite rows, `polyfit` on the finite columns — would level the scan and leave the gap absent rather than throw the image away. It changes what levelling **computes**, so it is a numerical task with its own ADR and golden update, not a validation tweak. |
| B-061 | **A rough opening radius of 0 means the estimate found nothing, and says so by returning a radius** | candidate — needs its own task | `estimate_rough_radius` can return 0 — `int(np.sqrt(median_area / np.pi))` is 0 for single-pixel objects, and without a scale the `min_size_px` floor is 0 too. `disk(0)` is a single pixel, so the opening is the identity, the "substrate" comes back equal to the image and `z_above` is zero everywhere. It looks like a result. M3-T13 deliberately left it legal: it is reachable on the unscaled noisy path **ADR-0025 measured and recorded**, so refusing it moves a number, and ADR-0010 keeps one intent to one commit. The fix is upstream — a rough estimate that found nothing should say so — and it needs a decision about what the fallback radius is. |
| B-054 | Optimise the committed README figures | candidate | `images/yolo_sam2_comparison.png` (3.2 MB) and `images/log.png` (3.0 MB) are the largest tracked non-notebook files and miss the 1 MB pre-commit limit planned in M1-T07. They are real content, so they were not untracked in M1-T01 — recompress or downscale instead, together with the README rewrite (M9-T01). |

---

## Rejected

| Item | Why |
|---|---|
| HTTP backend for the React client | The product is an offline desktop application. Building a second delivery channel doubles the surface with no user. See ADR-0007. |
| Cloud storage / multi-user projects | Single-operator, single-machine, offline-first. Projects are directories precisely so that sync tools are somebody else's problem. |
| Storing images or masks as SQLite blobs | Violates the storage rule: images stay files, the database stores metadata. Blobs would make projects opaque and unmergeable. |
| A dependency-injection framework | Explicit wiring in `app/` is enough at this size and stays greppable. |
| Rewriting the scientific core from scratch | It works, it is measured, and it is the reason the project exists. It gets moved and fixed, not replaced. |
