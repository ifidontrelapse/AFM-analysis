# ADR-0025 — An unknown AFM scale is not a fabricated 1.0

- **Status:** Accepted
- **Date:** 2026-08-06
- **Affects:** `nanoscope/infrastructure/storage/loaders.py`,
  `nanoscope/core/entities/image.py`,
  `nanoscope/core/science/preprocessing/substrate.py` · **M3-T20** (the other half of audit
  **D-07**)
- **Numerical impact:** **5 golden keys added, 0 values changed.** Every phantom has a scale,
  so nothing recorded moves; the new keys record the no-scale path. Where the size filter was
  doing work, losing the scale moves the substrate — `afm_sparse_low_snr` keeps 3351 objects
  instead of 17 and opens with radius 5 instead of 8.

## Context

`load_afm(fmt="npy")` invented the metadata the file does not carry:

```python
pixel_size_nm=pixel_size_nm or 1.0,
scan_size_nm=scan_size_nm or float(z.shape[0]),
```

Three defects in two lines.

**A fabricated scale is indistinguishable from a measured one.** Every `_nm` downstream —
`radii_nm`, `typical_radius_nm`, `height_nm`, every exported column — becomes a pixel count
wearing nanometre units, and no consumer can tell the difference. This is precisely the failure
mode the invariant in PROJECT_RULES §3 exists to prevent, and ADR-0019 rejected the same
substitution when it was proposed for the detectors.

**`or` swallows an explicit `0.0`.** A caller who passes `pixel_size_nm=0.0` — wrong, but
deliberate — is silently overruled with 1.0. Zero is not another spelling of "unknown"; it is a
caller error, and `PixelScale.__post_init__` has said so since M2-T02.

**`float(z.shape[0])` is not dimensionally a size.** A row count is used as a length in
nanometres.

ADR-0019 named this task as the completion of D-07 and fixed the order:

> *"This task makes `None` survivable; M3-T20 makes it honest. In that order, because the
> reverse order introduces a `None` into a path that still crashes on it."*

The detectors have handled `None` since M3-T11. The AFM preprocessing chain has not, because
until now `None` could not reach it.

## Decision

**`load_afm` passes through what it was given. Unknown is `None`, all the way to the entity.**

```python
def _given_and_positive(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if not value > 0:
        raise ValueError(f"{name} must be positive when given, got {value!r}")
    return value
```

`AFMRawData.pixel_size_nm`, `AFMRawData.scan_size_nm` and the two matching fields on
`PreprocessingResult` become `float | None`. The invariant has to hold at the entity, or the next
loader re-invents the default and nothing catches it.

**A given scale must be a real size.** `None` is a state; `0`, a negative number and `nan` are
caller errors, and each now raises with the parameter name and the value (PROJECT_RULES §3). The
rule is `PixelScale.__post_init__`'s, restated at the boundary where the number enters the
system — see *Alternatives* for why the value object itself is not adopted as the field type.

**Without a scale, the physical size filter is skipped — loudly.** `build_substrate_map` needs
the scale for three things: `radii_nm`, the `min_size_nm` filter (ADR-0024, one commit ago), and
`estimate_rough_radius`'s floor. None of the three can be expressed without it. So the nanometre
outputs come back `None`, the filter does not run, and the skip is logged at `WARNING` naming the
minimum it could not apply.

**An unscaled run is exactly a scaled run with `min_size_nm=0`**, in every pixel-space field —
that is the precise statement of what is dropped, and a test pins it.

**Loudly, because it is not free.** The morphological opening is pixel-space arithmetic, but the
*radius it uses* is derived from the filtered radii, so on an image where the filter was doing
work, losing it moves the substrate. Measured on `afm_sparse_low_snr`: **17 objects become
3351**, the typical radius falls from **2.99 px to 0.80**, and the opening radius goes **8 → 5**.
On the four phantoms where the filter removed nothing, the substrate is bit-identical.

That is D-04's mechanism arriving by a different road — a radius estimate driven by single-pixel
noise — which is exactly why the skip must announce itself. A size filter that is silently off
is the defect the previous commit spent an ADR removing. This one is off for a stated reason,
said out loud, once, at the site where it is dropped, and the warning names the consequence.

**The rough-radius floor becomes 0 px, not `min_size_nm` px.** Reading a nanometre number as a
pixel count when the conversion is unavailable is the unit confusion ADR-0024 deleted. Absent is
0, not "the same number in different units".

## Consequences

**Positive**

- D-07 is closed on both sides. An AFM scan without metadata reports what it knows and no more.
- `load_afm(fmt="npy", pixel_size_nm=0.0)` is now an error instead of a silent 1.0.
- The AFM branch and the SEM/TEM branch of `run_pipeline` now carry the same type,
  `float | None`, so the union that M3-T11 had to annotate by hand is the honest signature of
  both.
- **M3-T17 inherits a contract instead of a question.** When `_read_nanoscope_z` stops dividing
  `None` by `samps`, the state it produces already has a defined meaning everywhere downstream.

**Negative**

- **A user who relied on the fabricated 1.0 loses their nanometres.** They were never
  nanometres, but a script that read `typical_radius_nm` off an npy scan now reads `None`, and
  that is a breaking change with no compatibility path — deliberately, because the alternative is
  continuing to return a number that is wrong.
- **Losing the scale costs the radius estimate on a noisy scan**, quantified below: the filter
  feeds `typical_radius_px`, which feeds the opening radius, so the substrate is *not*
  scale-independent in general. It is identical on four of the five phantoms and different on the
  fifth, and the fifth is the one where the filter was load-bearing.
- **`sizes` is now a dict whose `_nm` values may be `None`.** It is typed `dict[str, Any]`
  (M2-T02 recorded why), so nothing static will catch a consumer that multiplies them. The tests
  and this ADR are the guard until `sizes` becomes a real type — M3-T14's neighbourhood.
- One more branch in `estimate_radius_otsu`, which is already the most-amended function in the
  package: D-01, D-05, D-06, D-10, D-04 and now this.

**Neutral**

- **No existing number moves.** Every phantom has a scale, so every recorded value is
  byte-identical; the new keys record a path that was previously unreachable from the loader.
- mypy is unchanged at 15. Widening two fields to `float | None` introduced no new error,
  because the consumers that were going to break had already been widened by M3-T11.
- `scan_size_nm` is not derived from `pixel_size_nm * z.shape[0]` — see *Alternatives*.

## The measured delta

**5 keys added, 0 values changed** — one `build_substrate_map_no_scale` subtree per AFM phantom.
Every phantom carries a scale, so nothing recorded moves; what is new is a path that could not be
reached from `load_afm` before this commit.

What that new subtree records, against the scaled run beside it:

| phantom | opening radius | objects kept | typical radius px | substrate |
|---|---|---|---|---|
| `afm_flat_monodisperse` | 19 → 19 | 24 → 24 | 7.2031 → 7.2031 | identical |
| `afm_coarse_pixels` | 11 → 11 | 14 → 14 | 4.0093 → 4.0093 | identical |
| `afm_dense_overlapping` | 16 → 16 | 51 → 51 | 6.0239 → 6.0239 | identical |
| `afm_tilted_polydisperse` | 18 → 18 | 29 → **30** | 6.8868 → 6.8520 | identical |
| `afm_sparse_low_snr` | 8 → **5** | 17 → **3351** | 2.9854 → **0.7979** | **differs** |

`radii_nm` is `None` on all five, which is the invariant this task exists for.

**The four clean phantoms cost nothing and the noisy one costs the substrate.** On
`afm_tilted_polydisperse` the filter was removing exactly one object, so the radius moves by
0.03 px and the opening radius does not move at all. On `afm_sparse_low_snr` the filter was
removing 3334 of 3351, and without it the median radius is a noise pixel's.

**This is the strongest argument in the ADR for the warning, and it was not visible until the
golden was regenerated.** The decision — skip, do not refuse — stands, because the alternative
throws away correct pixel-space work for a metadata gap. But "the pixel-space result is
unaffected" would have been a comfortable thing to write and it is false on one phantom in five.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Keep defaulting to `1.0` and document it | The documentation cannot reach the CSV a reader opens six months later. A number that is wrong in a column labelled `_nm` is worse than an absent one | The pipeline never reported physical units |
| Refuse to preprocess without a scale | Throws away pixel-space work that is entirely correct, and contradicts ADR-0019's ruling that unknown scale is a supported state. An npy file legitimately has no header | AFM output were only ever consumed in nanometres |
| Skip the size filter silently | This is D-04 with a different cause: a noise filter that is off and says nothing. The whole of ADR-0024 argues against exactly this | The filter were not load-bearing |
| Apply `min_size_nm` as pixels when the scale is missing | The unit confusion ADR-0024 deleted, re-entered through the back door. "5 nm" and "5 px" are the same number and different facts | The parameter were dimensionless |
| Derive `scan_size_nm = pixel_size_nm * z.shape[0]` | Dimensionally sound and still not done: the SPM path derives the pixel size from `samps` (**columns**), the line this ADR deletes used `z.shape[0]` (**rows**), and nothing in the codebase settles that convention for a non-square scan. Absent beats confidently wrong on one axis | A single axis convention were established (M3-T14) |
| Adopt `PixelScale` as the field type | Its guard is exactly this rule, but it does not model a *scan size*, and changing `AFMRawData.pixel_size_nm` from `float` to a value object ripples through four layers and changes what `dataclasses.asdict` produces — which the golden records. The `core/values` note assigns that to the task with a consumer for it | The entity boundary were being reworked anyway |
| Treat `0.0` as "unknown" too | Then a caller cannot express a mistake, and the loader keeps a second spelling for `None`. ADR-0019: "a second way to say unknown is a second thing to check" | Zero were a physically meaningful pixel size |

## Compliance

- `tests/unit/test_afm_io.py` — the npy characterization test flips from *invents a scale of one
  nm per pixel* to *reports an unknown scale*, plus a parametrised refusal of `0.0`, `-1.0` and
  `nan` for both parameters, and one test that a caller may know the pixel size without knowing
  the scan size.
- `tests/unit/test_unknown_scale.py::TestPreprocessingWithoutAScale` — 8 tests: the substrate,
  `z_above`, the opening radius and `radii_px` are **bit-identical** with and without a scale
  *where the filter removed nothing*; an unscaled run **equals a scaled run with
  `min_size_nm=0`**; on a noisy image it does **not** equal the default scaled run, and the
  opening radius differs; the `_nm` fields are `None`; the dropped filter is warned about and
  names the minimum; nothing is filtered away; the rough-radius floor is dropped rather than
  reinterpreted; and `run_preprocessing` on a bare npy file — the route the defect actually
  travels — reports `None` end to end.
- **Restoring `pixel_size_nm or 1.0` turns 6 red.**
- Golden: `build_substrate_map_no_scale` for all 5 AFM phantoms.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-07
- `ADR-0019` — the first half: unknown scale is a state, not a crash, and the note assigning this
  half to M3-T20
- `ADR-0024` — the filter this task has to decide the fate of, and the unit rule it must not undo
- **M3-T17** — the same state arriving from the SPM header, which this contract now defines
