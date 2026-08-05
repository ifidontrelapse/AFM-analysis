# ADR-0019 — An unknown pixel scale is a state, not a crash

- **Status:** Accepted
- **Date:** 2026-08-05
- **Affects:** `nanoscope/core/entities/detection.py` · `core/science/detection/{base,log}.py` ·
  `infrastructure/models/yolo.py` · `core/ports/detector.py` ·
  `application/use_cases/pipeline.py` · audit **D-07** · M3-T11
- **Numerical impact:** none. No recorded number moves; 168 golden keys are added, all of them
  behaviour that previously raised `TypeError` and so had nothing to record.

## Context

`MicroscopyData.nm_per_pixel` is `float | None`, and `run_pipeline` reads it into the variable
it passes to the detector:

```python
nm_per_pixel = data.nm_per_pixel        # None when the physical scale is unknown
detections = detector.detect(image, nm_per_pixel)
```

Both detectors then multiplied by it without looking:

```python
radius_nm = sigma_px * np.sqrt(2) * pixel_size_nm     # log.py
radius_nm = radius_px * pixel_size_nm                 # yolo.py
TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
```

So "scale unknown" — an explicitly supported state, typed as such on the entity — crashed the
one operation the pipeline exists to perform. It is reachable by the ordinary SEM/TEM route,
where a plain image file carries no scale metadata at all.

The invariant the audit states for D-07 is **a physical value is either physical or absent**:
never zero, never pixel-valued, never a crash. The project already keeps it in one place —
`measure_geometry_from_mask` returns `area_nm2=None, radius_nm=None` when `nm_per_pixel is
None`, and says so in its docstring. The detectors simply never got the same treatment.

## Decision

**`None` propagates, and the nanometre value becomes absent.** The pixel-space fields are
unaffected: a position and a radius in pixels do not need a scale to exist.

```python
Detection.radius_nm: float | None            # was: float
```

- `YoloDetector._boxes_to_detections` — `None if pixel_size_nm is None else radius_px * scale`.
- `detect_particles` — the `(N, 4)` blob array's fourth column is `NaN` throughout.
- `BaseDetector._blobs_to_detections` — maps that `NaN` to `None` on the entity.

**Why `NaN` in the array and `None` on the entity.** An ndarray column has no `None`: it is one
dtype for the whole column, and forcing `object` to carry a missing marker would change the
dtype of every recorded detection. `NaN` *is* the float convention for "no measurement", and it
is what `measure_all_baseline`'s DataFrame would coerce a `None` into anyway. The entity is not
an array, so it says the honest thing directly.

**This is not the `NaN` ADR-0018 removed, one commit earlier.** That one was *arithmetic* — the
product of dividing by zero — and it propagated into decisions: a threshold comparison, a sigma
range, a downstream `int()`. This one is a *marker* in a reporting column, written deliberately,
consumed by exactly one line that turns it into `None`, and never compared against anything. The
difference is not the value, it is whether anything downstream is allowed to compute with it.

**The variable in `run_pipeline` is now annotated `float | None`.** It was inferred as `float`
from the AFM branch, which is why mypy reported the SEM/TEM assignment on the next branch as an
error. That error *was* D-07's static footprint, sitting in the baseline since M1-T04 — mypy had
found this defect, at the assignment rather than at the crash, and nobody read it that way.

## Consequences

**Positive**

- SEM and TEM images without scale metadata run end to end. Before this, that path had exactly
  one outcome, and it was an exception.
- `radius_px` is still there, so a caller who knows the scale out of band can apply it later —
  the information is not destroyed, only not invented.
- mypy: **19 → 18** errors, and the one that went is the one that described this defect.

**Negative**

- `Detection.radius_nm` is now `float | None`, so every consumer must handle the `None`. There
  are two producers and, today, no consumer that arithmetics on it — but a future consumer must
  not write `d.radius_nm * x` unguarded. That is the cost of the invariant, and mypy enforces it
  in the strict layers.
- One more `NaN` exists on purpose, in a project that has spent two commits removing accidental
  ones. It is commented at the site with the distinction above.

**Neutral**

- No number moves. Every phantom has a scale, so every existing recorded value is
  byte-identical; the 168 new keys record a path that used to raise.

## What is deliberately not in this commit

- **`load_afm(fmt="npy")` fabricating a scale** (`pixel_size_nm or 1.0`) — that is **D-20 /
  M3-T20**, and it is the *reason* `None` rarely reaches the detectors from the AFM side today.
  This task makes `None` survivable; M3-T20 makes it honest. In that order, because the reverse
  order introduces a `None` into a path that still crashes on it.
- **`build_substrate_map` and the preprocessing chain**, which divide by the scale
  (`int(min_size_nm / pixel_size_nm)`). `AFMRawData.pixel_size_nm` is typed `float` and the
  loaders never produce `None` for it today; when M3-T20 changes that, the same invariant has to
  be applied there, with its own delta.
- **`plot_detections`**, which multiplies tick positions by the scale. Only the notebooks call
  it, always with an AFM scan.
- **`run_sam2_from_blobs`'s `if nm_per_pixel else None`**, which also swallows an explicit
  `0.0`. Wrong for the same family of reasons, in a module this task does not touch.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Default the scale to `1.0` when unknown | This is the defect the invariant exists to prevent: every `_nm` becomes a pixel count wearing nanometre units, and no consumer can tell. It is also exactly what M3-T20 has to undo elsewhere | — |
| Raise a clear error instead of a `TypeError` | Better than the status quo and still wrong: scale-unknown is a supported state, and SEM/TEM images legitimately arrive without metadata | The pipeline promised physical output for every modality |
| Keep `radius_nm: float` and use `NaN` on the entity too | Consistent with the array, but it puts a float that is not a number into a field typed as a number, and every consumer needs `math.isnan` instead of `is None`. Dataclasses can express absence; arrays cannot | `Detection` were itself stored column-wise |
| Refuse to build the `Detection` at all without a scale | Throws away correct pixel-space results because a piece of metadata is missing | Detections were only ever consumed in nm |

## Compliance

- `tests/unit/test_unknown_scale.py` — 8 tests: `detect_particles` no longer raises; the
  pixel-space columns are **bit-identical** with and without a scale; the nm column is `NaN`,
  not `radius_px`; `LogDetector` and `YoloDetector` both report `radius_nm is None`; a known
  scale still produces nanometres in both; and the SEM path through `run_pipeline` — the route
  D-07 actually travels — completes. Substituting the tempting wrong fix, `pixel_size_nm or
  1.0`, turns **4** red.
- Golden: `detect_particles_no_scale` for all 5 AFM phantoms (with the resulting `Detection`
  count that carries a radius — **0**), and `boxes_to_detections_{scaled,no_scale}` for all 7
  phantoms that carry the YOLO block.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-07
- `ADR-0018` — the accidental `NaN` this one is careful not to be
- `nanoscope/core/science/measurement/geometry.py` — the same invariant, kept since M2-T06
- **M3-T20** — the loader that fabricates a scale, and why it comes second
