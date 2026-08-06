# CURRENT TASK

**ID:** `M3-T20`
**Title:** An AFM scan without a scale is a state, not a fabricated 1.0
**Milestone:** M3 — Numerical correctness, ninth task
**Defect:** **M3-T20** (high), found by the M1-T06 tests · **ADR:** **ADR-0025**
**Branch:** `sci/npy-no-invented-scale` (stacked on `sci/min-size-in-nm`)
**Status:** planned — no code written yet.

---

## Why this task is next

It is the other half of **D-07**. M3-T11 made `None` survivable in the detectors and said so:

> *"`load_afm(fmt="npy")` fabricating a scale — that is M3-T20, and it is the reason `None`
> rarely reaches the detectors from the AFM side today. This task makes `None` survivable;
> M3-T20 makes it honest. In that order, because the reverse order introduces a `None` into a
> path that still crashes on it."* — ADR-0019

The path is now survivable, so the fabrication can go.

```python
pixel_size_nm=pixel_size_nm or 1.0,
scan_size_nm=scan_size_nm or float(z.shape[0]),
```

Three defects in two lines:

1. **A fabricated scale is indistinguishable from a real one.** Every downstream `_nm` becomes a
   pixel count wearing nanometre units, and no consumer can tell — the exact failure the
   invariant in PROJECT_RULES §3 exists to prevent.
2. **`or` swallows an explicit `0.0`.** A caller who says `pixel_size_nm=0.0` — wrong, but
   deliberate — is silently overruled with 1.0 rather than corrected.
3. **`float(z.shape[0])` is not dimensionally a size.** A row count is used as a length in
   nanometres.

---

## Scope

**In scope**

1. `load_afm(fmt="npy")` passes through what it was given. Unknown stays `None`; a value that is
   given must be positive, or it is a `ValueError` naming the parameter and its value
2. `AFMRawData.pixel_size_nm` and `.scan_size_nm` become `float | None`, and
   `PreprocessingResult` with them — the invariant has to hold at the entity, or the next loader
   re-invents the default
3. **The preprocessing chain accepts `None`**, which ADR-0019 assigned here explicitly: without a
   scale there are no `radii_nm`, and `min_size_nm` cannot be applied. That is a **decision**,
   and the ADR carries it: no scale → no physical filter, **warned**, never silent
4. The harness records the no-scale preprocessing path, as M3-T11 did for the detectors
5. **ADR-0025**

**Out of scope**

- **M3-T17** — `_read_nanoscope_z` dividing `None` by `samps` when the header has no `Scan Size`.
  Same state arriving from the other loader, and it is its own task and commit. This task defines
  the contract that one will satisfy
- **Deriving `scan_size_nm` from `pixel_size_nm * z.shape[0]`.** Dimensionally sound, and still
  not done here: the SPM path derives the pixel size from `samps` (**columns**), the old npy line
  used `z.shape[0]` (**rows**), and nothing in the codebase settles that axis convention for a
  non-square scan. Absent is better than confidently wrong on one axis
- **Adopting `PixelScale` as the field type.** Its positivity guard is the rule this task
  enforces at the boundary, but changing `AFMRawData.pixel_size_nm` from `float` to a value
  object ripples through four layers and changes what `dataclasses.asdict` produces
- `run_sam2_from_blobs`'s `if nm_per_pixel else None`, the same `or` family in a module this
  task does not touch (ADR-0019 already filed it)

---

## The decision this task has to make

`build_substrate_map` needs the scale for three things: `radii_nm`, the `min_size_nm` filter
(M3-T02, one commit ago), and `estimate_rough_radius`'s floor. With no scale, none of the three
can be expressed. The options:

| | |
|---|---|
| **Refuse** — raise, an AFM scan must have a scale | Throws away pixel-space work that is correct, and contradicts ADR-0019's ruling that unknown scale is a supported state |
| **Skip the filter silently** | Re-creates D-04 exactly: a noise filter that is off and says nothing |
| **Skip the filter, loudly** ✅ | The substrate estimate is pixel-space and unaffected; the physical threshold is inapplicable, and the log says so at the moment it is dropped |

Chosen: the third. `radii_nm` and `typical_radius_nm` come back `None`; `radii_px` and the
opening radius are bit-identical to a scaled run, which is the property the tests pin.

---

## Definition of done

- [ ] No fabricated scale anywhere in `load_afm`; an explicit non-positive value raises
- [ ] `float | None` on both entities, and mypy strict is happy at the boundary
- [ ] `build_substrate_map` returns pixel-space results with `None` in every `_nm` field
- [ ] The dropped filter is warned, once, naming the value it could not apply
- [ ] Tests, including: pixel-space output identical with and without a scale
- [ ] `make check` green; delta quantified (expect **keys added, values unchanged** — every
      phantom has a scale)
- [ ] ADR-0025; `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [ ] Commit: `M3-T20: an unknown AFM scale is not a fabricated 1.0`

---

## Notes

`tests/unit/test_afm_io.py::test_npy_without_metadata_invents_a_scale_of_one_nm_per_pixel`
characterizes the defect and says in its docstring that its assertions flip to `None` when this
task lands. It is the test to rewrite first.
