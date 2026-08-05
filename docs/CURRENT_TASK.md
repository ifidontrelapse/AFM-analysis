# CURRENT TASK

**ID:** `M3-T03`
**Title:** YOLO input — normalise, *then* cast
**Milestone:** M3 — Numerical correctness, second task
**Defect:** **D-03**, critical · **ADR:** **ADR-0015**
**Branch:** `sci/yolo-normalise-then-cast` (`sci/` — the branch intentionally changes
scientific output, PROJECT_RULES §7)
**Status:** **done 2026-08-05.** Rewritten for `M3-T04` at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

> **Read this before reading the numbers as good news.** The fix makes YOLO's input equal to
> the data. It does not make YOLO's output better, and nothing in the gate can say whether it
> did: the weights in `checkpoints/best12x.pt` were trained on images the *old* path produced.
> That question belongs to **M3-T15** (evaluation) and **M7** (retraining).

---

## Why this task was next

`_prepare_image` cast a float height map in nanometres to `uint8` and normalised afterwards.
Every YOLO detection this repository has ever produced came from an image that had already
lost between 19% and 97% of its levels — and, on maps taller than 255 nm, had wrapped. It is
the last critical defect that needed no operator decision: the correct order is not a
question of physics.

---

## Scope

**In scope**

1. `_prepare_image`: normalise in float, cast after — the three lines both backends share
2. The characterization golden, re-baselined and the delta quantified
3. `tests/unit/test_yolo_input.py` — properties of the mapping height → grey level
4. **ADR-0015**, and the rounding-versus-truncation choice inside it

**Out of scope**

- **D-21 / M3-T04** — the anisotropic resize two lines above. Same function, separate defect,
  separate commit (ADR-0010)
- **D-09 / M3-T05** — YOLO confidence never reaching `Detection`
- **M3-T18** — `_last_result` typed `None` and dereferenced unguarded (mypy)
- Retraining, and any claim about detection quality

---

## Definition of done

- [x] `_prepare_image` normalises before casting; both backends fixed by the same change
- [x] A test suite that fails on the old order — **5 of 6 red** when it is restored
- [x] Golden regenerated, delta quantified: **67 differences, all under
      `yolo_input_preparation`**, on all 7 phantoms; nothing else in the baseline moved
- [x] ADR-0015 written, including what the fix does *not* claim
- [x] `make check` green
- [ ] CI green — pushed at the end of the session; CI is the run that matters, since it is
      the environment without torch
- [x] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md`, `PROJECT_CONTEXT.md` updated
- [x] Commit: `M3-T03: normalise the YOLO input before casting it to uint8`

---

## The delta

| phantom | range (nm) | levels before | after | corr(before, after) |
|---|---|---:|---:|---:|
| `afm_flat_monodisperse` | −0.5 … 18.1 | 19 | 256 | 0.997 |
| `afm_tilted_polydisperse` | −1.8 … 45.6 | 47 | 255 | 0.914 |
| `afm_dense_overlapping` | −0.6 … 19.0 | 19 | 256 | 0.997 |
| `afm_sparse_low_snr` | −4.3 … 5.0 | **8** | 239 | **−0.499** |
| `afm_coarse_pixels` | −0.6 … 20.0 | 21 | 256 | 0.997 |
| `sem_bright_particles` | 14.5 … 230.6 | 208 | 255 | 1.000 |
| `tem_dark_particles` | 23.7 … 234.3 | 200 | 254 | 1.000 |

The audit measured 12.6% retention on one realistic map. The phantoms give the shape of it:
**the cleaner the sample, the worse the corruption**, because a narrow height range has
fewer integers inside it. The one negative correlation is the point at which "lossy" stops
being the right word.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Reading the green as "detections improved" | Stated in the ADR, in `Progress.md`, in `STATE.md` and here. Inference is outside the gate by §6; the claim is that the input is now the data |
| The fix is bundled with the aspect-ratio defect in the same three lines | It is not. D-21 is M3-T04, its own branch and ADR |
| The golden's D-03 fields (`mean_abs_diff_vs_normalize_first`) become meaningless once the defect is gone | They become the opposite: the reference block now asserts an invariant of 0.0. Reordering the cast moves it on all 7 phantoms |
| A `uint8` cast decision (round vs truncate) smuggled in silently | Named in the ADR with its reason: truncation matches the harness reference, so the defect's measuring stick reads exactly 0.0 |

---

## Notes for the next session

**M3-T04 / D-21** is the natural next task — same file, same three lines, and the other half
of "the YOLO path is fed correctly": `cv2.resize` squashes a non-square scan to 640×640, and
`_scale_boxes` stretches the boxes back with two different factors.

**The two remaining critical defects are not engineering-blocked, they are decision-blocked.**
B2 (D-04, `min_size_nm` floors to zero on 90% of real scans) and B3 (D-12, TEM finds 0 of 22
particles) have been open since M0. M3 cannot close without them.

Carried, still not tasks: **B-058** (the golden compares CPython exception text — ADR before
any Python upgrade), **B-054** (two README figures over 1 MB, M9-T01).
