# STATE

**Last updated:** 2026-08-06 · **Branch:** `sci/spm-header-without-scan-size` · **Base commit:** `aceb5c7`

> This file is mandatory and must be updated at the end of **every** development session.
> Read it first when a session starts.

---

## Current milestone

**M3 — Numerical correctness**

Fix the defects the audit reproduced. **The rules change here:** every task gets its own
commit, its own ADR, its own golden update, and a quantified before/after delta in
`docs/Progress.md`. Never bundled (ADR-0010).

**M2 closed 2026-08-04** — sixteen tasks, 2 021 lines of science moved into four named
layers, and **not one number changed**. Five of six exit criteria met in full; the sixth
(ports) is partly met on purpose and M4 owns the rest. Milestone summary in
`docs/Progress.md`.

**M1 closed 2026-08-04** — all eleven tasks done, four of five exit criteria met. The
fifth (no tracked file over 1 MB) has two known exceptions, the README figures, filed as
**B-054** and deferred to M9-T01. Milestone summary in `docs/Progress.md`.

## Current task

**None selected.** **All four `critical` defects are closed**, **D-07 is closed on all three of
its faces**, and so are all five operator decisions that were open when M3 started. The next task
is **M3-T12** — D-08, the last unblocked `high` one: empty measurements must return a
schema-stable DataFrame. **B6 → M3-T16** is the last operator answer waiting to be executed, and
**B-040** goes last of everything because it rewrites every SHA above it.

**`M3-T17` done 2026-08-06 (ADR-0026)** — the SPM parser's `else` branch, written to tolerate a
header with no `Scan Size`, divided `None` by `samps` on the very next line: **the fallback
crashed on the branch it had just taken**. It now returns `(None, None, z)` — the height map
decodes as always, only the metadata is absent, and ADR-0025 gave that state a meaning everywhere
downstream one commit earlier. The same expression's other failure mode went with it
(`Samps/line: 0` was a `ZeroDivisionError` naming nothing), and a *stated* non-positive
`Scan Size` is rejected as well: **absent and wrong are different**. Delta: **0 golden
differences, and none was possible** — `afm_io` has no phantom, so its 28 unit tests are the whole
safety net. **mypy 15 → 14**, the removed error being this defect's own (`-> np.ndarray` on a
function returning a three-tuple). 3 tests; restoring the division turns 3 red.

**`M3-T20` done 2026-08-06 (ADR-0025)** — the npy loader no longer invents `1.0` nm/px and a scan
size equal to the row count. `None` is unknown and passes through to the entity; a scale that
*is* given must be positive, so `0.0`, `-1` and `nan` raise instead of being swallowed by `or`.
`build_substrate_map` accepts `None`: the `_nm` outputs are absent and the `min_size_nm` filter
cannot be applied, which is **warned**. Delta: **5 golden keys added, 0 values changed** — every
phantom has a scale. The new keys carry the finding: an unscaled run is exactly a scaled run with
`min_size_nm=0`, so on four phantoms the substrate is bit-identical and on `afm_sparse_low_snr`
it is not — **17 objects become 3351**, the typical radius falls 2.99 px → 0.80 and the opening
radius 8 → 5. **Losing the scale is losing the filter**, which is D-04's mechanism one commit
after D-04 was closed. 10 tests; restoring `pixel_size_nm or 1.0` turns 6 red.

**`M3-T02` done 2026-08-06 (ADR-0024, decision B2)** — **D-04 fixed, the last `critical`.**
`int(min_size_nm / pixel_size_nm)` is gone; the filter compares `radii_nm >= min_size_nm`, so a
physical minimum stays physical. Delta: **47 differences — 27 changed, 15 added, 5 removed**.
`afm_sparse_low_snr` drops **75 objects to 17** and everything derived from its radii moves with
it; the other four AFM phantoms are byte-identical and **no measured height moves anywhere**,
because the final opening radius is 8 on both sides. **Re-measuring all 628 scan headers**
reproduces the audit's 90 % (568/628) and adds what it did not measure: the zero threshold cost
**nothing** on the 365 scans (58 %) coarser than 8.86 nm/px, where one pixel is already 5.5 nm;
it disabled the filter on the 203 (32 %) in the 5–8.86 band; and the finest 60 (10 %) were hurt
by **truncation**, not by the floor — `afm_sparse_low_snr`'s 2.5 px threshold became 2 px, and 58
of its 75 "objects" were noise living in that half-pixel. **mypy unchanged at 15**: a unit error
has no static shadow, which is why the suffix convention is the only check this class has.
5 tests; restoring the `int()` turns 3 red.

**`M3-T07` done 2026-08-05 (ADR-0018)** — D-11 fixed: `z_above / z_above.max()` at two sites
never checked its divisor. A flat map made every pixel `nan` and the code blamed the threshold;
a negative map flipped the topography and produced an adaptive threshold of **2.4997** against a
`[0, 1]`-normalised response. Both sites now stop on a non-positive or `nan` maximum —
`DEFAULT_THRESHOLD = 0.05` from the estimator, an empty `(0, 4)` from `detect_particles`.
**65 golden keys added, 0 changed**: the working path is byte-identical, and the wrong number
had never been recorded at all, because the harness wrote every scalar down as the string
`"non-array"` and its only negative degenerate input was *constant*. Fixing that is the larger
half of the commit.

**`M3-T06` done 2026-08-05 (ADR-0017)** — D-05/D-06 fixed: the empty-after-filter case raised
nothing and returned `nan`; it now raises with the parameter, its value and the largest object
measured, and `n_objects` counts survivors. **8 golden differences**, of which the headline is
`n_objects_reported` **1023 → 75** on `afm_sparse_low_snr` — a 13.6× over-count of
single-pixel noise. It also broke a test written in M3-T01 that had been passing *because* of
the `nan`.

**`M3-T04` done 2026-08-05 (ADR-0016)** — D-21 fixed: the scan is scaled isotropically and
padded to the model square instead of squashed into it, and `_scale_boxes` inverts exactly
that. **0 golden differences, 7 keys added**: a square scan is byte-identical, and every
phantom is square — which is why the harness gained `non_square_half_height` in the same
commit. Reading for it turned up **M3-T21**: `use_tiling=True`, the default, produces exactly
one crop, so the tiled backend has never tiled.

**`M3-T03` done 2026-08-05 (ADR-0015)** — D-03 fixed: the cast to `uint8` now happens
*after* the normalisation. **67 golden differences, all under `yolo_input_preparation`**;
grey levels reaching the network went from 8–208 to 239–256. The retention spread is
**3.1%–81.2%**, and the cleaner the scan the worse the loss: the quiet 5 nm phantom kept 8
levels of 256 and came out **anti-correlated** (−0.499) with a correctly prepared image.
**This does not mean detections improved** — the weights were trained on images the old path
produced; see the ADR's Consequences.

**All five operator decisions were answered on 2026-08-05** — B2, B3, B4, B6, B7 — and four are
executed: B4/M3-T09, B7/M3-T21, B3/M3-T10 and B2/M3-T02. **B6/M3-T16** is the one left, plus
**B-040** (purging `node_modules` and the weights from git history), which goes last because it
rewrites every SHA. **B-058** is done (ADR-0022).
**B-054** is closed: the operator deferred the two oversized README figures to the M9-T01 rewrite.

---

## Completed

### M3 — Numerical correctness (in progress)

- **M3-T17** ✅ (2026-08-06, **ADR-0026**) — **D-07's third face, and the last.** The SPM
  fallback for a header with no `Scan Size` set `scan_size_nm = None` and divided by `samps` on
  the next line. Now `(None, None, z)`: absent metadata, intact height map. `Samps/line: 0` and a
  stated `Scan Size: 0` raise instead, each naming its field — ADR-0025's absent-versus-wrong
  distinction, applied to the second loader so the two agree. **0 golden differences, and none
  was possible**: `afm_io` has no phantom, and that is recorded rather than left blank.
  **mypy 15 → 14** — the annotation `-> np.ndarray` on a function returning a three-tuple was
  this defect's static shadow, in the baseline since M1-T04. 3 tests; restoring the division
  turns 3 red.

- **M3-T20** ✅ (2026-08-06, **ADR-0025**) — **the other half of D-07.** `load_afm(fmt="npy")`
  fabricated `pixel_size_nm=1.0` and `scan_size_nm=float(z.shape[0])`, so every `_nm` downstream
  was a pixel count wearing nanometre units and no consumer could tell — and `or` swallowed an
  explicit `0.0` on the way. The loader now passes through what it was given, `AFMRawData` and
  `PreprocessingResult` carry `float | None`, and a scale that *is* given must be positive.
  `build_substrate_map` takes `None`: `radii_nm` and `typical_radius_nm` are absent, and the
  `min_size_nm` filter is skipped **with a warning** — silently would be D-04 again. Delta:
  **5 keys added, 0 values changed**; the new `build_substrate_map_no_scale` records that an
  unscaled run equals a scaled one with `min_size_nm=0` — bit-identical substrate on four
  phantoms, and on `afm_sparse_low_snr` **17 objects → 3351**, typical radius **2.99 → 0.80 px**,
  opening radius **8 → 5**. The ADR's draft claimed the pixel-space result was unaffected; the
  golden disagreed on one phantom in five and the ADR now says what was measured. **M3-T17
  inherits a contract instead of a question.** 10 tests; restoring the fabrication turns 6 red.

- **M3-T02** ✅ (2026-08-06, **ADR-0024**, decision **B2**) — **D-04 fixed: the minimum particle
  size is a physical size.** `min_size_nm` was converted to pixels with `int()` at three sites,
  compared against `radii_px`, and then — three lines later, twice, identically — `radii_px` was
  converted back to nanometres for the result. The comparison now happens where the parameter was
  always stated: `radii_nm >= min_size_nm`. B2 answered **filter in nanometres**; a "floor of at
  least 1 px", the other candidate, was rejected because at 29.3 nm/px it would discard
  everything under 29.3 nm. Delta: **47 differences — 27 changed, 15 added, 5 removed**;
  `afm_sparse_low_snr` **75 → 17** objects, the other four AFM phantoms **byte-identical**, and
  **no height moves anywhere** (the final opening radius is 8 before and after, so `substrate`
  and `z_above` are unchanged). **The phantom built for D-04 does not move**, which is the
  finding: at 9.77 nm/px a single pixel is already 5.5 nm, so the broken filter and the correct
  one agree. Re-read of all **628** scan headers: 90 % floored to zero (568), of which **365
  (58 %) had nothing to remove**, **203 (32 %)** lost a working noise filter, and the finest
  **60 (10 %)** were hurt by truncation rather than the floor. **mypy unchanged at 15** — a unit
  error is invisible to a type checker. 5 tests; restoring the `int()` turns 3 red. The
  duplicated `radii_nm` assignment went with it: the change forced the line to move.

- **M3-T10** ✅ (2026-08-05, **ADR-0023**, decision **B3**) — **D-12 fixed: TEM finds 22 of 22
  where it found 0.** Both detectors kept the bright side unconditionally — the LoG one by
  thresholding and by `blob_log` itself, the YOLO one by inverting every image because the weights
  expect dark particles — and TEM images by absorption, so both were working on the background.
  B3 answered **configured, with a per-modality default**: an auto-detector's failure mode is
  D-12's own (zero particles, no error), and the operator could not tell a bad guess from an empty
  sample. `Polarity`, written in M2-T02 for this task and adopted by nothing since, is now a
  `PipelineConfig` field whose `None` resolves to the modality's convention in `run_pipeline`.
  **One inversion at the entrance**, `max - z`: its own inverse, and positive-maximum-safe per
  ADR-0018. Both detectors in one commit, because it is one defect mirrored. Delta: **19 values
  changed, 12 keys added** — `tem_dark_particles` 0 → 22 blobs, its prepared YOLO input 43.3 →
  211.7 mean grey, `config_fields` 12 → 13; **SEM and all five AFM phantoms byte-identical**.
  **Not claimed:** better YOLO detections — inference is outside the gate, so what is shown is
  that the input is right. 14 tests.

- **B-058** ✅ (2026-08-05, **ADR-0022**) — the golden compared exception messages exactly, and
  most of them are CPython's or a library's. 3.14 reworded `too many values to unpack` and the
  first real CI run called it characterization drift (M1-T08). Now the **type** and the **raising
  function** are always compared and the **message** only when we wrote it — the frame must be
  inside `nanoscope` *and* the raising line must be an explicit `raise`, because either alone
  misclassifies: `h, w = z.shape` in our file is CPython's wording, and skimage raises explicitly
  too. **15 keys renamed to `error_message_unchecked`** (skipped by `compare`, still recorded),
  **7 remain compared, all `estimate_radius_otsu`'s** — the ones PROJECT_RULES §3 governs. 0
  values changed. **A Python upgrade no longer reads as drift**, which `STATE.md` listed as the
  precondition for touching the interpreter. 6 tests.

- **M3-T21** ✅ (2026-08-05, **ADR-0021**, decision **B7**) — the tiled YOLO backend **has never
  tiled**: `_prepare_image` emits one 640 px square and the crop shape is 640, so `get_crops_xy`
  computes one step per axis. It ran the direct backend's work through an extra library, more
  slowly, and small particles were never seen at native resolution — the only reason tiling
  exists. The overlap cannot rescue it (`int((side−shape)/step)+1` is 1 for any step when
  `side == shape`, asserted at 0/25/50/75 %); only input size can, and real tiling needs 1120 px.
  B7 answered **keep it, stop defaulting to it**: `use_tiling=False` in both `YoloDetector` and
  `PipelineConfig`, and the degenerate case now says so in the log. Delta: **zero golden
  difference** — inference is outside the gate — but the backends are **not bit-identical** even
  at one crop (`CombineDetections` adds a second NMS), so real detections may move and **no claim
  is made that either is better**; M3-T15 owns that question, and until it exists nobody can
  choose between "upsample to 1120" and "smaller crops". 9 tests.

- **M3-T09** ✅ (2026-08-05, **ADR-0020**, decision **B4**) — **D-10 fixed**. `disk(8.5)` is an
  18x18 element with no centre pixel, so the opening was biased by half a pixel; three sites fed
  it a float and each did something different. The operator answered B4 **round up**, and the
  `ceil` lives in `get_substrate_map` — the funnel all three pass through — so one line fixes
  them all. Up rather than down because a radius smaller than a particle recovers a "substrate"
  containing the particle's own top, while an over-large disk only over-smooths, which the method
  already tolerates. `build_substrate_map` reports the integer it used, keeping ADR-0014's
  principle. Delta: **696 golden values move, 0 keys added** — radius +1 or +2 on all five AFM
  phantoms, **no particle count changes**, largest height move **0.049 nm (0.37 %)** on
  `afm_dense_overlapping`. The 696 are propagation, not magnitude. **mypy 18 → 15**, and the
  three that went were this defect's static shadow, in the baseline since M1-T04. 11 tests;
  restoring the floor turns 4 red.

- **M3-T11** ✅ (2026-08-05, **ADR-0019**) — **D-07 fixed**. `MicroscopyData.nm_per_pixel` is
  `float | None`, `run_pipeline` passes it to the detector unread, and both detectors multiplied
  by it: `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`. An SEM or TEM
  image without scale metadata had exactly one outcome, and it was an exception. Now `None`
  propagates and the physical value becomes **absent**, which is the invariant D-07 states and
  the one `measure_geometry_from_mask` has kept since M2-T06. `Detection.radius_nm` is
  `float | None`; the blob array's nm column is `NaN`, because one ndarray column is one dtype,
  and `_blobs_to_detections` converts it at the entity boundary. **That NaN is not the NaN
  ADR-0018 removed one commit earlier** — this one is a marker in a reporting column, read by one
  line and never computed with; that one came out of arithmetic and reached a threshold
  comparison. Delta: **168 golden keys added, 0 changed** — every phantom has a scale, so nothing
  recorded moves. **mypy 19 → 18, and the error that went was the defect**: `pipeline.py:62` had
  been reporting D-07 at the assignment rather than at the crash since M1-T04. 8 tests;
  substituting `pixel_size_nm or 1.0` — the tempting wrong fix, and what the npy loader does
  today — turns 4 red.

- **M3-T07** ✅ (2026-08-05, **ADR-0018**) — **D-11 fixed**. The LoG path normalises with
  `z_above / z_above.max()` in two places and checked the divisor in neither. `max() == 0` gives
  a wholly `nan` image, `blob_log` finds nothing, and the operator is told to *lower the
  threshold* — a knob that cannot help. `max() < 0` **inverts the topography**, so the substrate
  outshines the peaks; measured on caps at −10 nm the adaptive threshold came out **2.4997**,
  compared against responses that live in `[0, 1]`. The guard is `not z_max > 0`, **not**
  `z_max <= 0`, because the two differ exactly on `nan` and `nan` is the case that spreads.
  **Zero particles is the answer, not an error** — the opposite call from ADR-0017 one commit
  earlier, because there the *caller* asked the impossible and here the *data* is simply flat,
  which is a legitimate scan region. Delta: **65 golden keys added, 0 changed** — every phantom
  goes through `build_substrate_map`, which guarantees `z_above >= 0`, so nothing on the working
  path moves. The negative case is reachable only via `LogDetector.detect` on raw SEM/TEM, which
  is **D-12**, still on B3. **The harness could not see this defect at all** and two changes in
  the same commit fixed that: `negative_with_structure` (the old `all_negative` is *constant*, so
  the flip has nothing to flip) and recording scalars instead of the string `"non-array"` — that
  line is why 2.4997 sat unrecorded since Phase 0. 11 tests; restoring the raw division turns 3
  red, and the two that stay green do so by construction: a `nan` image also yields no blobs,
  which is precisely how the defect survived.

- **M3-T06** ✅ (2026-08-05, **ADR-0017**) — **D-05 and D-06 fixed**, one commit because they
  are the same eight lines. The size filter could remove every object, and then `np.median([])`
  returned `nan` with a warning; the `nan` reached the LoG sigma range and failed two calls
  later as `zero-size array to reduction operation minimum`. It now raises where it happens,
  naming the parameter, its value **and the largest object measured** — without that third
  number, "no particles here" and "your minimum is 100× too large" read identically. And
  `n_objects` counts survivors instead of the pre-filter population. Delta: **8 golden
  differences** — `n_objects_reported` **1023 → 75** on `afm_sparse_low_snr` (13.6× over-count
  of single-pixel noise), the `extreme_aspect` degenerate input now fails as
  `estimate_radius_otsu` instead of `cannot convert float NaN to integer` one call downstream,
  plus 5 added keys recording D-05's own reproduction. **Only one phantom moved, and why is the
  point:** D-04 floors `min_size_pixel` to 0 on coarse scans, so the filter usually removes
  nothing — this fix starts mattering on real data the day **B2** is answered. 4 tests;
  restoring the old behaviour turns 3 red. **It also turned an M3-T01 test red**, one that had
  been passing because the sizing silently returned `nan` into a field it never read.

- **M3-T04** ✅ (2026-08-05, **ADR-0016**) — **D-21 fixed**: `_prepare_image` squashed every
  scan into a 640 × 640 square and `_scale_boxes` stretched the boxes back per axis. The two
  agreed, so boxes landed correctly — the defect is that **on a 2:1 scan the model saw
  ellipses**, and `radius_px = min(w, h) / 2` reported the smaller half-axis as a radius. Now
  one isotropic scale, a border of 255 (what the lowest point looks like after the inversion,
  so it reads as substrate rather than as an edge), applied *after* the normalisation, and one
  helper shared by the forward and inverse maps so they cannot drift. Delta: **0 golden
  differences, 7 keys added** — square scans are byte-identical and every phantom is square,
  so the harness gained `non_square_half_height` to characterize the path at all. 5 geometry
  tests; restoring the squash turns 4 red. **Found while reading, filed not fixed: M3-T21.**

- **M3-T03** ✅ (2026-08-05, **ADR-0015**) — **D-03 fixed**: `_prepare_image` cast the float
  height map to `uint8` *before* normalising, keeping only whichever integers 0…255 fell
  inside the map's range and wrapping the rest, then stretching the survivors so the result
  looked correctly exposed. Delta: **67 golden differences, all under
  `yolo_input_preparation`** across all 7 phantoms; unique grey levels **8–208 → 239–256**,
  retention **3.1%–81.2% → 100%**. The spread is the finding — **the cleaner the scan, the
  worse the loss** — and on the quiet 5 nm phantom the old image is **anti-correlated**
  (−0.499) with the correct one, so this was never merely a resolution defect. The cast
  truncates rather than rounds, matching the harness's own reference, which drops
  `mean_abs_diff_vs_normalize_first` to **0.0** and turns the field Phase 0 added to size
  the defect into a permanent guard. 6 tests; restoring the order turns 5 red.
  **Not claimed: better detections.** The weights were trained on images the old path
  produced, and inference is outside the gate (§6) — M3-T15 and M7 own that question.

- **M3-T01** ✅ (2026-08-04, **ADR-0014**) — **D-01 fixed**: `build_substrate_map`'s
  manual-radius branch raised `UnboundLocalError` on **100% of calls** since it was
  written, because the shared `return` read a variable only the other branch bound. The fix
  is one line — `opening_radius = manual_radius_px` — and it deliberately applies **no
  rounding and no floor**: both would pre-empt open decision B4 (M3-T09) or silently
  override an explicit request. Delta: **50 golden differences, every one under
  `build_substrate_map_manual`**; the automatic path is untouched. The harness now records
  the branch's returned arrays instead of only its failure — otherwise the fix would have
  left it less characterized than while broken. 6 tests; restoring the bug turns 5 red.

### M2 — Domain extraction ✅ (closed 2026-08-04)

- **M2-T01** ✅ (2026-08-04) — `nanoscope/` exists: the six layers from ADR-0011
  (`app` `core` `application` `infrastructure` `gui` `resources`) plus `py.typed`, each
  `__init__` stating that layer's half of the dependency rule. Distribution renamed
  `afm-analysis` → `nanoscope`. The regenerated `uv.lock` was **diffed package by package
  before committing** — 119 shared packages, **0 version changes** — because CI runs
  `uv sync --locked` and a quiet re-resolution of numpy or scipy would move the golden for
  a reason unrelated to the task. mypy now checks 20 files instead of 13, and the strict
  `nanoscope.*` override that M1-T04 wrote before the package existed **binds for the
  first time**: 0 errors, strict from line one. **Zero code moved**; no sub-package below
  the layer level, since each arrives with its content in M2-T02…T08.

- **M2-T02** ✅ (2026-08-04) — the first scientific code to move, in **three commits** so
  that drift would be attributable without bisecting. The six dataclasses left
  `src/types.py` for `nanoscope/core/entities/`; `src/types.py` is now a shim that defines
  nothing, verified by loading the pre-move module beside the new one — identical fields,
  order, defaults and factories, and `src.types.X is nanoscope.core.entities.X` for all
  six. **One `Detection` class in the process, not two.** The strict `nanoscope.*` override
  then caught three things verbatim legacy code does not satisfy: two bare generics
  tightened to `dict[str, Any]`, and `Detection.bbox` given a scoped `type: ignore` —
  mypy complaining there *is* **D-16**, and fixing it moves a number the golden records, so
  M3 owns it; `warn_unused_ignores` makes the ignore expire itself. **nanoscope: 0 mypy
  errors.** Finally `Modality`, `Polarity`, `PixelScale`, `DeviceKind` with 8
  mutation-validated tests — **defined, adopted by nothing**, because adoption changes what
  `asdict` produces. Golden: **zero drift**.

- **M2-T03** ✅ (2026-08-04) — preprocessing moved to
  `nanoscope/core/science/preprocessing/` (`flatten.py` + `substrate.py`); `preprocess.py`
  is a shim. The first move of real behaviour — plane fitting, line detrending,
  morphological opening, Otsu. **Proved before the gate ran:** all six functions
  AST-identical, docstrings differing only in trailing whitespace, and the 5 mypy errors
  travelled with the code (21 before, 21 after). Golden: **zero drift**. What the task
  actually settled is how legacy enters a strict package: **declared once in configuration**
  — mypy at default strictness for `nanoscope.core.science.*`, ruff still blocking there
  but ignoring six named rules — instead of a `type: ignore` on every audited defect across
  fifteen more moves. Every entry names the task that deletes it (M2-T11, M2-T12, M3).

- **M2-T04…T06** ✅ (2026-08-04) — three tasks on one branch (they share shims), **16
  definitions moved, golden zero drift**. I/O split along parsing-versus-the-world:
  SPM decoding to `core/science/io/`, the path-opening functions to
  `infrastructure/storage/`. The LoG detector and its ABC to `core/science/detection/` —
  all 7 definitions AST-identical, and `detect_particles` is recorded for all 8 phantoms.
  Measurement split AFM height from mask geometry, which is the point of M2-T06: the
  modality-neutral code was trapped in an AFM module, so the SEM/TEM path depended on
  `src.measure` by accident. Four more `src/` modules are shims. **The `ImageLoader` port
  was deliberately not written** — M2-T08 defines the ports wholesale. mypy 21 → 21.
  Three of ruff's safe fixes landed in `loaders.py` and are named, not rounded up to
  "verbatim". **`RUF046` was wrong about the science**: `round(np.float64)` is not an int,
  so obeying it would have changed the dtype of every measurement DataFrame's `x_px`
  column — it is now ignored with that reason attached.

- **M2-T07 / M2-T08** ✅ (2026-08-04) — the model-backed code left the domain:
  `YoloDetector` → `infrastructure/models/`, the SAM2 runners beside it, and
  `afm_to_rgb`/`overlay_masks` → `infrastructure/imaging/` (neither ever belonged to
  SAM2). **Nothing under `core` imports torch, ultralytics, sam2 or patched_yolo_infer any
  more** — the dependency rule is now a fact, and a test asserts it against `sys.modules`.
  `F821` caught two dangling references the split created, before any test ran. mypy 21 →
  21, after both moved modules joined `core.science` at default strictness. **M2-T08 was
  narrowed on purpose: one port, not seven.** `Detector` is satisfied today by
  `LogDetector` and `YoloDetector` from opposite layers; the other six have no
  implementation and no caller, so they ship with their first adapter, and
  `core/ports/__init__.py` carries the table naming the task for each.

- **M2-T09 / M2-T10** ✅ (2026-08-04) — the layout became enforceable and the rules became
  executable. **All five import cycles (D-18) had one cause**: `src/__init__.py` re-exported
  the pipeline, and Python runs a package `__init__` first, so importing the "dependency
  root" loaded SAM2 and matplotlib. Nothing ever used `from src import X` — emptying one
  file fixed all five. `import src.types` **1198 → 187 modules, 0.77 s → 0.07 s**;
  `nanoscope.core.entities` **626 → 185**, pandas moved behind `TYPE_CHECKING`.
  `test_import_graph.py` checks direction statically over the AST and weight dynamically in
  a subprocess; both proven to fail. **The M2 exit criterion "< 100 modules" was
  unachievable** — numpy alone is 141 — and is corrected in `Roadmap.md` to a named
  heavy-import assertion plus a 250 bound. M2-T10 put the capability matrix in
  `application/capabilities.py` and **fixed D-14**: validation now runs before any detector
  is constructed, with byte-identical messages. 12 tests carry it, because the golden never
  calls `run_pipeline`.

- **M2-T11…T14** ✅ (2026-08-04) — the library stopped printing, started speaking English,
  shed four dead functions and became installable. **Zero numbers moved**, and for the first
  time that took a *declared* golden re-baseline: 6 changed lines, none of them numeric —
  4 translated exception messages plus `stdout_lines` 8→0 and 4→0, because the golden
  records how much a function prints. **It also caught a bug in M2-T11 before any human
  did** (`"1%%"` is only an escape when `logging` formats, which it does not without args).
  **No `LogSink` port — ADR-0013**: it would only wrap `logging`, whose `Handler` is already
  the extension point. That is the second of seven planned ports to dissolve on contact with
  reality. **M2-T13 deleted 4 of the audit's 10 "unreachable" functions and kept 6** —
  `estimate_log_threshold` is recorded by the golden, `load_microscopy_image` is the only
  SEM/TEM entry point, three are used by the notebooks. `nanoscope` is now a real wheel
  (`py.typed` in, `src/` out) and the `pythonpath` hack is half deleted. Ruff findings inside
  `nanoscope/` with ignores off: **64 → 13**.

- **M2-T15 / M2-T16** ✅ (2026-08-04) — **`src/` deleted entirely**, and the milestone with
  it. The title understated the task: three modules had never had a shim and had to move
  first (`pipeline` and `preprocessing_pipeline` → `application/use_cases/`, `visualization`
  → `infrastructure/imaging/`). `pythonpath` deleted outright; mypy points at one package.
  A test caught a naming trap a review would not have: a module and a function of the same
  name shadow each other through `__init__`. M2-T16 rewrote `PROJECT_CONTEXT.md`, which had
  drifted to describing `src/`, the deleted frontend and a `pytest.ini` removed in M1-T05.

### M1 — Repository hygiene ✅ (closed 2026-08-04)

- **M1-T01** ✅ (2026-08-03) — tracked files 2 877 → **77**; `frontend/node_modules`
  (2 800 files) untracked; `yolov8s-world.pt` (26 MB) removed from the index before it
  entered history; `.gitignore` rewritten; `.claude/settings.json` now shared; junk
  deleted (`.zip`, four `__pycache__/`, tool caches, empty `output/` and `notebooks/`,
  stray root `package-lock.json`); `plan.md` archived to `docs/archive/`.
  Characterization: **zero drift**.
- **M1-T11** ✅ — absorbed into M1-T01.
- **M1-T02** ✅ (2026-08-03) — pytest 9.1.1, pytest-cov 7.1.0, ruff 0.16.1, mypy 2.3.0
  declared and installed; no runtime version moved, golden still stable. Baseline
  measured: **196 ruff findings** (109 in `src/`), **30 mypy errors**, **1 test, failing**.
  Nothing fixed — that is M1-T03/T04 and M2.
- **M1-T03** ✅ (2026-08-03) — ruff configuration repaired: `fix = true` removed (it made
  `ruff check` rewrite sources), `select`/`ignore` moved under `[tool.ruff.lint]`,
  py311 → py312, template `known-first-party` fixed, dead `S101` dropped, notebooks
  excluded from lint. `src/` findings unchanged at 109 — a repair, not a rule change.
  Total 196 → 128. Characterization: **zero drift**.
- **M1-T04** ✅ (2026-08-04) — mypy configured: strict for `nanoscope.*` from its first
  line; `src/` checked but **not** silenced (22 errors after per-module stub handling).
  All 30 default errors classified before writing config: 13 statically confirm audit
  defects **D-01, D-02, D-07, D-10, D-16**, and **3 new defects** were found and filed
  (**M3-T17…T19**), including a crash in the SPM parser's no-`Scan Size` fallback.
- **M1-T05** ✅ (2026-08-04) — the characterization golden now runs under `pytest`, via a
  single new seam in `capture.py` (`diff_against_golden()`); the CLI is unchanged. Marked
  `slow` (**192 s measured**, not the ~100 s the docs claimed); `pytest -m "not slow"`
  skips it in 1.4 s. `pytest.ini` folded into `pyproject.toml` and deleted — while it
  existed, pytest ignored `[tool.pytest.ini_options]` silently. The negative case was
  proven, not assumed: a perturbed golden produced a red run naming the moved quantity.
  **The M2 safety net is now mechanical.**
- **M1-T06** ✅ (2026-08-04) — `tests/test_io.py` (no assertions, wrong exception, absent
  fixture path) deleted; replaced by `tests/unit/test_afm_io.py`: **22 tests** over a
  synthetic Nanoscope byte stream derived from a real local header — round trip,
  calibration, unit conversion, 8 failure modes, npy and SEM/TEM. No binary fixture, no
  `data/`. **`pytest` is green for the first time (23 passed, 200 s).** The suite was
  validated by mutation: 4 edits to the parser, 3 killed immediately, and the 4th exposed a
  test that could not fail — now fixed. One new defect found → **M3-T20**.
- **M1-T07** ✅ (2026-08-04) — pre-commit: **9 hooks, each demonstrated failing** on a
  deliberately bad staged file. ruff runs as a `repo: local` hook on the project's own
  version, so no second version is ever declared. Rewriting hooks (format, whitespace) skip
  `src/` **and `preprocess_batch.py`** — the `--all-files` sweep caught them editing the
  scientific core, which the original `^src/` exclusion missed; refusing hooks apply
  everywhere. pytest and mypy stay off the commit path by design. `src/` files modified:
  **0**; golden: zero drift.
- **M1-T08** ✅ (2026-08-04) — CI written and verified locally: format → lint → tests+golden,
  `src/` reported not blocking. CI installs a `ci` group with **no torch, ultralytics, sam2
  or patched-yolo-infer** — every heavy import in `src/` turned out to be function-local —
  and a step fails the job if one appears. Two traps caught by running it: `uv run` re-syncs
  and would have reinstalled the full runtime (`UV_NO_SYNC` set), and `ruff format` rewrites
  Python inside Markdown docs (`*.md` excluded). The legacy exclusion moved into
  `pyproject.toml`, declared once for hooks and CI. Both rejection cases confirmed red.
  **Then it was pushed, and three runs found what local verification could not:** no
  readable failure reason (job logs need admin → diagnostics added), a non-existent
  `setup-uv@v9` tag (my error; both actions now pinned exactly), and — the real one — a
  single golden difference that was an exception *message*, not a number. CI resolved
  **Python 3.14**, which reworded `too many values to unpack`; 3.12 is now pinned and
  asserted. **Run 4 is green.** The underlying fragility — the golden stores CPython
  exception text — is filed as **B-058** and needs an ADR, not a quiet edit.
- **M1-T09** ✅ (2026-08-04) — notebook outputs stripped with the configured hook:
  **8.3 MB → 32 KB**, every one of the 45 code cells intact, and the outputs remain in git
  history. Both notebooks moved to `notebooks/` with a README stating they are experiments,
  that nothing may import them, and how to recover the outputs. `main.ipynb` — a tracked
  **0-byte file that was not valid JSON** (audit §330) — deleted. Tracked working tree
  17 MB → **7.8 MB**. **`pre-commit run --all-files` is green for the first time**; the
  last red was a missing final newline in one archived document.
- **M1-T10** ✅ (2026-08-04) — one gate, one description: a 53-line `Makefile`
  (`check` `format` `lint` `test` `fast` `golden` `types` `lint-legacy`; bare `make` lists
  them), and **CI rewritten to call the targets** — the workflow no longer holds a copy of
  any command, which was the point. Proven to fail closed: a misformatted file stopped
  `check` at step 1 in **0.04 s**, exit 2, never reaching the 190 s test step; a failing
  test failed its target. `types`/`lint-legacy` stay outside `check` because the legacy
  baseline is non-zero by design — a gate that cannot pass is a gate people skip. Writing
  it exposed that the three existing descriptions had already drifted: `PROJECT_RULES` §6
  listed `mypy nanoscope` (no such package yet) and a golden command M1-T05 had folded
  into `pytest`. **CI run 14 green on the first try, 216 s**, environment assertion intact.
  **M1 closes here.**

### Decisions executed (2026-08-04)

- **B1 → `nanoscope`** — ADR-0011 Accepted. Unblocks every M2 task.
- **B5 → delete** — **ADR-0012** (supersedes ADR-0007): `frontend/` and
  `preprocess_batch.py` removed. Tracked files **78 → 63**, and the blocking lint/format
  carve-out shrank from two paths to one, `src/`, which M2 then dissolves. Ruff findings in
  the legacy core **117 → 109**, all now in `src/`. Both files remain in git history.

### M0 — Engineering foundation (2026-08-03)

- Repository analysed: 12 source modules / 2 021 LOC, plus a React client, notebooks and
  an existing Phase 0 audit
- Strengths and weaknesses recorded with evidence → `docs/Architecture.md` §2
- Target Clean Architecture defined (`core` / `application` / `infrastructure` / `gui` / `app`)
- Project constitution written → `docs/PROJECT_RULES.md`
- 10 milestones, 110 tasks → `docs/Roadmap.md`, `docs/TASKS.md`
- 11 ADRs written → `docs/ADR/`
- Session/state protocol established → this file, `docs/Progress.md`, `docs/CURRENT_TASK.md`
- First task selected → `M1-T01`

### Inherited from earlier work (pre-M0, already in the repository)

- Working scientific pipeline: SPM I/O, flattening, substrate estimation, LoG and YOLO
  detection, SAM2 segmentation, height measurement
- Phase 0 audit with 24 reproduced defects → `docs/audit/2026-07-28-baseline-audit.md`
- Characterization golden baseline with 8 seeded phantoms →
  `docs/audit/characterization-baseline.md`, `tests/characterization/`

---

## In progress

**M2 is closed and M3 is well under way.** M3-T01, T02, T03, T04, T06, T07, T09, T10, T11 and
T21 are done, plus B-058 — eleven defects closed in three sessions. **All four `critical`
defects are closed** (D-01, D-03, D-04, D-12), and with M3-T02 the last of them is the one that
had been open since M0 waiting on a decision.
**The YOLO input path is now correct in three respects** — the data survives preparation, the
sample keeps its shape, and the polarity matches the modality — and none of those claims extends
to detection quality, which nothing in the gate can measure. **The LoG path no longer constructs
a `nan` image**, and its adaptive threshold now always lands in the interval it is compared
against. **The noise filter runs for the first time** on the scans where it was floored away.

**Repository state:** `main` is at `aceb5c7` and carries all of M0, M1, M2 and M3-T01.
Twelve branches stack in order: `sci/yolo-normalise-then-cast` (M3-T03) →
`sci/yolo-letterbox` (M3-T04) → `sci/otsu-sizing` (M3-T06) → `sci/log-zero-max` (M3-T07) →
`sci/unknown-scale` (M3-T11) → `sci/opening-radius-ceil` (M3-T09) → `sci/tiling-default`
(M3-T21) → `sci/golden-exception-text` (B-058) → `sci/detection-polarity` (M3-T10) →
`sci/min-size-in-nm` (M3-T02) → `sci/npy-no-invented-scale` (M3-T20) →
`sci/spm-header-without-scan-size` (M3-T17). **All eleven are pushed and all eleven are
green on CI** — the seven that had never been pushed ran as #44–#50 on 2026-08-06, 139–459 s
each, and M3-T20 followed as **#52**, every one `success`. CI on `main` is
green: **216 s**, of which
`make test` is 194 s, and the environment assertion (Python 3.12 + CPU-only) passes, so the
green is green for the right reason.

**There is no `src/`.** One package, `nanoscope`, 41 modules across four layers, installed
rather than path-hacked.

**Legacy in transit is declared, not hidden.** `nanoscope.core.science.*` runs at mypy's
default strictness and carries six named ruff ignores; every entry names the task that
deletes it (M2-T11, M2-T12, M3). The rest of `nanoscope` stays strict and 0.

Locally, `make check` is green end to end: format, lint, then the full suite including the
golden, exit 0.

---

## Blocked / needs decision

Decisions only the operator can make. Each blocks a specific task.

| # | Question | Blocks | Why it needs the operator |
|---|---|---|---|
| B6 | **Real sample data in git.** `data/` holds 628 SPM scans and is ignored. Should one small representative scan be committed as a test fixture? | M3-T16 | Data ownership and repository size |

**Answered 2026-08-05 by the operator, and all now executed:**

- **B4 → round up.** M3-T09, ADR-0020.
- **B7 → keep the tiled backend, stop defaulting to it.** M3-T21, ADR-0021.
- **B3 → polarity is configured, with a per-modality default.** M3-T10, ADR-0023.
- **B2 → filter in nanometres, delete the `int()`.** M3-T02, ADR-0024. The floor-of-1-px
  alternative was rejected in the ADR: at 29.3 nm/px it discards everything under 29.3 nm.

**Closed 2026-08-04 by the operator:**

- **B1 — package name → `nanoscope`.** ADR-0011 moves from Proposed to **Accepted**. This
  was the last thing blocking **M2**; M2-T01 can start as soon as M1 closes.
- **B5 — fate of the parked work → delete.** `frontend/` (21 tracked files) and
  `preprocess_batch.py` removed under **ADR-0012**, which supersedes ADR-0007. The third
  part of B5, the notebooks, was answered differently in M1-T09: kept, stripped, moved.

None of the remaining questions blocks M1 or M2.

---

## Next

1. **Execute M3-T12** — D-08, the last unblocked `high` one. Rewrite `docs/CURRENT_TASK.md` for
   it first. After that the `medium` ones in order: T05, T08, T13, T14
2. **Every `critical` defect is closed and every decision except B6 is answered and executed.**
   What remains in M3 is engineering, in severity order: T20, T12, T17 are the `high` ones, then
   T05, T08, T13, T14. **B6 → M3-T16** is the last operator answer waiting; **B-040** goes last
   of all, because it rewrites every SHA above it
3. `make types` joins `make check` as blocking — the one deviation recorded against M1's
   exit criteria. `src/` is gone, so the only thing left is the 20 errors that arrived
   inside the moved science; they belong to M3 and M2-T12
4. **B-058 is done (ADR-0022)** — a Python upgrade no longer reads as drift, so the 3.12 pin in
   CI is now a choice rather than a constraint
5. **B-054** (two README figures over 1 MB) is the one M1 exit criterion left open;
   it belongs to the README rewrite in M9-T01

---

## Health indicators

| Indicator | Value | Target | Source |
|---|---|---|---|
| Tracked files | **105** (was 2 854) | see note | `git ls-files \| wc -l` |
| Tracked working tree | **7.6 MB** ✅ (was 17 MB) | — | `git ls-files -z \| xargs -0 du -ch` |
| Tracked model weights | **0** ✅ (was 1) | 0 | `git ls-files '*.pt'` |
| `.git` size | 81 MB | — | `du -sh .git` — history unchanged, see B-040 |
| Library LOC | 2 021 | — | `wc -l nanoscope/**/*.py` |
| Meaningful tests | **218, all passing** ✅ (was 1, failing) | ≥ 80% of core | `pytest -q` |
| Golden enforced automatically | **yes** ✅ (was: by discipline) | yes | `pytest` |
| `src/` modules moved into `nanoscope/` | **12 of 12** ✅ — `src/` deleted | 12 | `git ls-files` |
| ruff findings, declared-and-owned | **14** in `nanoscope/` (was 109 in `src/`) | 0 | `make lint-legacy` |
| ruff findings, blocking | **0** ✅ | 0 | `make lint` |
| mypy errors | **14**, all inherited with moved code, none silenced; new code strict | 0 | `make types` |
| Characterization phantoms | 8 (7 carry `yolo_input_preparation`) | 8 | `tests/characterization/` |
| Open defects | **20** (was 28) — D-01, D-03, D-21, D-05, D-06, D-11, D-07, D-10, D-12 closed; M3-T21 opened | 0 critical | audit §2, M3-T17…T21 |
| Import cycles | **0** ✅ (was 5), and a test refuses new ones | 0 | `tests/unit/test_import_graph.py` |
| `print` calls in library code | **0** ✅ (was 13), asserted per module | 0 | `tests/unit/test_logging.py` |
| Non-English lines in library code | **0** ✅ (was 197) | 0 | `grep -rn "[а-яА-ЯёЁ]"` |
| Lint/type/test gate | **green end to end** ✅ — hooks on commit, CI on push | stays green | GitHub Actions |
| The gate has one definition | **yes** ✅ — `make check`, and CI calls the same targets | one | `Makefile` |
| Tracked files over 1 MB | **2** ❌ — two README figures, B-054 | 0 | `git ls-files` + `ls -l` |

> **The `< 100` target has done its job and expired — the count passed it at M2-T07.** It was M1's measure of
> *junk* — 2 800 `node_modules` files. M2 adds real source: each move leaves a shim and
> creates two or three modules. Passing 100 means the extraction is working, not that
> hygiene regressed. The
> meaningful successor is the row above it: **tracked files over 1 MB**, which must stay
> at zero once B-054 closes.
| Commit-time gate | **9 hooks, all proven to fire** ✅ (was: none) | enforced | `pre-commit run` |
