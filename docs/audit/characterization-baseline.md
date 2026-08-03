# Characterization baseline

**Date:** 2026-07-28 · **Commit:** `11e0ecc` · **Golden file:** `tests/characterization/golden/baseline.json`

This is the safety net for the Phase 2 refactor. It records what the code **does today**, not
what it should do. Several recorded numbers are scientifically wrong — the audit says which and
why. The point is to make *accidental* change impossible to miss, so that *intentional* change
stands out and has to be justified.

> **The rule.** A refactor must leave every number here unchanged. If a number moves, either the
> refactor has a bug, or the change was intentional — in which case it needs an ADR, a changelog
> entry, and a golden update **in the same commit**.

---

## 1. Usage

```bash
python tests/characterization/capture.py            # compare against the golden file; exit 1 on drift
python tests/characterization/capture.py --write    # regenerate after a declared change
```

The comparison reports a path-addressed diff, so drift points at the exact quantity that moved:

```
afm_flat_monodisperse.log_detection.detect_particles_p20.n_blobs: 24 -> 23
afm_flat_monodisperse.baseline_measurement.measure_all_baseline.col::height_nm.mean: 17.94 -> 18.02
```

**Tolerance:** `rtol=1e-6`, `atol=1e-9` for floats; counts, dtypes, error types and error
messages must match exactly. The tolerance is far tighter than any scientifically meaningful
difference — it exists to catch accidents, not to certify physical accuracy.

**Runtime:** ~100 s single-threaded, CPU only. No model weights, no GPU, no network, no file I/O
outside the golden file.

---

## 2. What is covered

### Fixtures (`tests/characterization/phantoms.py`)

Deterministic, seeded, generated in-process — no binary test data enters git. Each carries its
ground truth (`centres_yx_px`, `radii_px`, `heights_nm`), so the same module becomes the backbone
of the Phase 6 evaluation harness rather than being throwaway audit scaffolding.

| Phantom | Purpose |
|---|---|
| `afm_flat_monodisperse` | Ideal case. If numbers move here, the change is real. |
| `afm_tilted_polydisperse` | Plane tilt + per-line offsets + mixed sizes. Exercises flattening. |
| `afm_dense_overlapping` | Touching particles. Stresses LoG overlap and ring baselines. |
| `afm_sparse_low_snr` | SNR ≈ 3. Where threshold selection breaks. |
| `afm_coarse_pixels` | `pixel_size_nm` 9.77 — the median of the operator's real scans. Pins defect D-04. |
| `sem_bright_particles` | Bright-on-dark, 8-bit. |
| `tem_dark_particles` | **Dark-on-bright.** The polarity counter-example. |
| `degenerate_inputs` | empty, 1×1, constant, all-negative, NaN, Inf, 1-D, 3-D, 2×4096. |

### Stages captured

Per AFM phantom: `flatten_plane` → `flatten_lines` (plus an idempotence probe) →
`build_substrate_map` (substrate, `z_above`, opening radius and type, full Otsu size statistics,
and the resolved `min_size_pixel`) → `estimate_log_params` / both threshold estimators →
`detect_particles` at percentiles 10/20/40 → `measure_all_baseline` (per-column digests, plus a
constant-height-offset invariance probe) → YOLO input preparation.

Arrays are stored as **order-independent digests** — shape, dtype, min/max/mean/std/sum and the
10/25/50/75/90 percentiles — so a refactor that legitimately reorders results does not read as a
numerical change, while any change in the values themselves still does.

Failures are recorded as first-class golden values (`error_type`, `error_message`, raising
function). Several inputs are *supposed* to fail; **how** they fail is part of the contract
Phase 2 will change deliberately.

---

## 3. Recorded behaviour worth knowing

These are measured values from the golden file, presented because they are the numbers most
likely to move.

### 3.1 Detection counts vs ground truth

| Phantom | true | Otsu `n_objects` | radii kept | LoG blobs (p20) | baseline rows |
|---|---:|---:|---:|---:|---:|
| `afm_flat_monodisperse` | 24 | 24 | 24 | **24** | 24 |
| `afm_tilted_polydisperse` | 30 | 29 | 29 | **30** | 30 |
| `afm_dense_overlapping` | 70 | 51 | 51 | **59** | 59 |
| `afm_sparse_low_snr` | 6 | **1023** | 75 | **0** | 0 |
| `afm_coarse_pixels` | 14 | 14 | 14 | **14** | 14 |

The pipeline is accurate on clean, well-separated fields and degrades predictably as density
rises. The low-SNR row is the interesting one: Otsu reports **1023 objects for 6 real particles**,
the noise-dominated radius estimate collapses to 2.88 px, and LoG then finds **nothing** — a
silent total failure reported to the user as "particles not found, try lowering the threshold".

That row also pins **D-06** unambiguously: 1023 reported vs 75 retained after size filtering.

### 3.2 TEM detects nothing — defect D-12

| Phantom | true particles | blobs found by the `run_pipeline` SEM/TEM path |
|---|---:|---:|
| `sem_bright_particles` | 22 | **22** |
| `tem_dark_particles` | 22 | **0** |

Same geometry, same count, inverted contrast. The LoG path keeps the bright side of the Otsu
threshold, so on conventional dark-on-bright TEM it characterises the background. TEM is one of
the three first-class modalities and currently returns zero particles.

### 3.3 YOLO input preparation — defect D-03

Unique grey levels reaching the model, current code vs `normalize → cast`:

| Phantom | current | correct | retained |
|---|---:|---:|---:|
| `afm_flat_monodisperse` | 19 | 256 | 7.4 % |
| `afm_tilted_polydisperse` | 47 | 255 | 18.4 % |
| `afm_dense_overlapping` | 19 | 256 | 7.4 % |
| `afm_sparse_low_snr` | 8 | 239 | 3.3 % |
| `afm_coarse_pixels` | 21 | 256 | 8.2 % |

Under 20% of the available dynamic range survives on every phantom. Fixing this **will** change
every YOLO detection — that is the intent, and it is why the fix needs its own commit and ADR.

### 3.4 `min_size_pixel` — defect D-04

`afm_coarse_pixels` (9.77 nm/px, the real-data median) records `min_size_pixel_used: 0` against
`2` for the 2 nm/px phantoms. The filter is off, exactly as it is on 90% of the operator's scans.

### 3.5 Degenerate inputs — the current, inconsistent contract

| Input | `flatten_plane` | `flatten_lines` | `build_substrate_map` | `detect_particles` |
|---|---|---|---|---|
| `empty` | ok | ok | ValueError | ValueError |
| `single_pixel` | ok | LinAlgError | ValueError | ok |
| `constant_zero` | ok | ok | ValueError | ok |
| `constant_nonzero` | ok | ok | ValueError | ok |
| `all_negative` | ok | ok | ValueError | ok |
| `with_nan` | ValueError | **ok** | ValueError | ok |
| `with_inf` | ValueError | **ok** | ValueError | ok |
| `one_dimensional` | ValueError | IndexError | TypeError | **ok** |
| `three_dimensional` | ValueError | ValueError | RuntimeError | **ok** |
| `extreme_aspect` | ok | ok | ValueError | ok |

Two things stand out. `flatten_lines` **propagates NaN and Inf without complaint** — only
`flatten_plane` catches them, and only because SciPy's least-squares refuses. And
`detect_particles` **accepts 1-D and 3-D arrays**: `blob_log` supports n-dimensional input, so an
`(8, 8, 3)` image is silently treated as a 3-D volume rather than rejected.

Five different exception types across ten malformed inputs, none naming the offending parameter.
Phase 2 item 7 replaces this table with one typed error taxonomy; the table is the "before".

### 3.6 Serialization boundary

```
PipelineResult -> json.dumps  ->  TypeError: Object of type ndarray is not JSON serializable
Detection().bbox              ->  ()   (length 0, annotated tuple[int, int, int, int])
```

---

## 4. Determinism

Verified: `--write` followed by a compare run reports no drift. Every source of randomness in the
fixtures is seeded through `np.random.default_rng(seed)`; `blob_log`, Otsu, and the morphological
operations are deterministic.

The environment is recorded in the golden file's `_meta` block:

| | |
|---|---|
| Python | 3.12.13 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| scikit-image | 0.26.0 |

**Upgrading scikit-image or SciPy may legitimately move these numbers.** Such a move is a
dependency-driven change, not a refactor bug: re-baseline it in its own commit with the version
bump, and record the delta.

---

## 5. Known gaps

Stated plainly so they are not mistaken for coverage:

- **No YOLO or SAM2 inference.** Only YOLO *input preparation* is characterised. Weights exist
  locally (`best12x.pt` 137 MB, `sam2.1_hiera_base_plus.pt` 324 MB) but model inference is not
  reproducible enough to be a golden. Segmentation therefore has **no safety net** — the Phase 2
  refactor of `segmentation.py` is unprotected, which is an argument for touching it last.
- **No committed real sample.** `data/` is gitignored and contains the operator's scan data
  (628 SPM files). The audit read 120 of them to measure the pixel-scale distribution, but
  committing any of them is the operator's call, not mine — see the question in the Phase 0
  report. The `afm_coarse_pixels` phantom is calibrated to the real median as a stand-in.
- **`_read_nanoscope_z` is not characterised.** It needs real binary input. All 120 files sampled
  parsed successfully, but the regex assumptions are unverified against other Nanoscope versions.
- **Frontend is not covered.** No component or contract test exists yet.

---

## 6. Files

```
tests/characterization/
├── phantoms.py            # deterministic fixtures + ground truth (no src/ dependency)
├── capture.py             # capture/compare runner
└── golden/baseline.json   # 103 KB, committed
```
