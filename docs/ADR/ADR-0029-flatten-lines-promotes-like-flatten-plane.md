# ADR-0029 — `flatten_lines` promotes the way `flatten_plane` does

- **Status:** Accepted
- **Date:** 2026-08-07
- **Affects:** `nanoscope/core/science/preprocessing/flatten.py` · audit **D-13** · M3-T08
- **Numerical impact:** **13 golden differences — 8 dtypes, 4 sums, 1 new group.** No float64
  value moves, so every phantom chain is byte-identical. On the newly recorded 8-bit input the
  levelled map was wrong by up to **257** — an integer output does not truncate a negative
  residual, it **wraps** it.

## Context

```python
result = np.empty_like(z)                      # keeps the input's dtype
for i, row in enumerate(z):
    coeffs = np.polyfit(xi, row, poly_order)
    result[i] = row - np.polyval(coeffs, xi)   # float64 residual, truncated on assignment
```

`np.polyfit` and `np.polyval` compute in float64 whatever they are handed, and the quantity being
stored is a *residual* — what is left of a row after its own best fit is removed. It is fractional
by construction. Writing it into an array that cannot hold fractions rounds every value toward
zero.

For an integer input that is not an approximation, it is a total loss. Measured on a `uint8` ramp:
the residuals reach **0.368** and the returned array is **all zeros**. The audit measured its own
ramp and got the same answer in the same shape — correct max 0.5625, actual all zeros.

**Boolean input is worse, and the audit did not measure it.** `result[i] = <float array>` into a
`bool` array stores `!= 0`, so a levelled map comes back as a **mask of where the residual was
non-zero**: max 1.0 where the real residual maximum is 0.45. A function documented to return
topography returns a boolean pattern.

**`flatten_plane` never had the defect.** It returns `z - plane` with `plane` float64 and lets
NumPy promote. So the two halves of "flattening" disagreed about the dtype of the same map:

```python
>>> flatten_plane(z_uint8).dtype, flatten_lines(z_uint8).dtype
(dtype('float64'), dtype('uint8'))
```

### Why this was invisible

The documented chain is `flatten_lines(flatten_plane(z))`, and `flatten_plane` promotes, so
`flatten_lines` receives float64 and behaves. Every phantom in the golden takes that route. The
defect lives on the routes where `flatten_lines` is called with an array that came from somewhere
else:

- **`load_microscopy_image` returns `uint8`** — `cv2.imread(..., IMREAD_GRAYSCALE)` — and it is
  the only file entry point SEM/TEM has.
- **`load_afm(fmt="npy")`** passes through whatever the `.npy` file holds.
- `README.md`, `project.md` and `PROJECT_CONTEXT.md` all document `flatten_lines` as a function a
  caller may use on its own, which the notebooks do.

## Decision

**The output dtype is `np.promote_types(z.dtype, np.float64)` — `flatten_plane`'s own rule,
written out.**

```python
result = np.empty_like(z, dtype=np.promote_types(z.dtype, np.float64))
```

Integers and booleans become float64. float32 becomes float64. float64 stays float64, which is why
nothing in the golden's five AFM chains moves. A wider float stays wide.

**Why not `dtype=np.float64` unconditionally.** For every dtype `np.polyfit` accepts today the two
are the same expression, so this is a choice about which rule is being stated, not about a value.
`flatten_plane` promotes; a hardcoded float64 would agree with it by coincidence and diverge on the
first input where promotion is not float64 — `longdouble`, which `flatten_plane` preserves. One
rule, in both halves of flattening, is the thing D-13 is about.

**float32 in becomes float64 out, and that is declared drift.** It is what `flatten_plane` already
does with a float32 input, and what NumPy would do if the line were written `z - trend` rather than
assigned into a pre-allocated buffer. The alternative — preserve float32, promote only integers —
fixes the reported defect and leaves the two functions disagreeing for the one dtype the SPM
loader produces most often.

## Consequences

**Positive**

- Levelling an 8-bit image returns the levelling. The SEM/TEM path had a preprocessing step that
  silently did nothing, and "returns zeros" is indistinguishable from "was already flat".
- `flatten_plane` and `flatten_lines` can be composed in either order without a dtype surprise.
- A boolean input is no longer answered with a mask that has the shape and name of topography.

**Negative**

- A float32 map costs twice the memory after levelling. For a 512×512 scan that is 1 MB against
  0.5 MB; for the largest scan the loaders can produce it is not a constraint the project has.
- Two functions now depend on the same promotion rule being spelled the same way. The test that
  holds them together is the only thing keeping that true.

**Neutral**

- No float64 value moves anywhere. Every recorded phantom chain is float64 by the time
  `flatten_lines` sees it.
- `poly_order`, the fit, the iteration and the shape are untouched.

## What is deliberately not in this commit

- **Typed input validation.** `np.promote_types` raises its own `TypeError` on a string array one
  line earlier than `np.polyfit` used to, and neither message names the parameter. Non-numeric
  dtypes, 1-D input, 3-D input and NaN belong to **M3-T13**, which takes every numerical entry
  point in one pass; fixing one of them here would be a taxonomy of one.
- **`flatten_plane`.** It is correct, and touching it would put a second intent in the commit.
- **B-059** — `nan <= 0` in `measure_all_baseline`. A different defect in a different file
  (ADR-0010).

## The measured delta

**13 differences: 8 dtype changes, 4 sums, and one added group.**

| Where | What moved |
|---|---|
| `degenerate_inputs.<8 inputs>.flatten_lines.result.dtype` | `float32` → `float64`. Eight of the eleven degenerate inputs level successfully; the other three raise, and **all three raise exactly what they raised before** — `IndexError` in `flatten_lines`, `LinAlgError` from lstsq, `ValueError` in `polyval` |
| `…extreme_aspect…result.sum` | `-3.92e-06` → `2.68e-13` |
| `…negative_with_structure…result.sum` | `1.52e-06` → `2.46e-12` |
| `…with_nan…result.sum` · `…with_inf…result.sum` | `-4.12e-07` → `-3.42e-14` · `2.07e-07` → `4.88e-15` |
| `flatten_dtypes` | added: five input dtypes × (input digest, `flatten_plane`, `flatten_lines`) |

**The four sums are the fix visible as a physical property.** A least-squares residual sums to
zero over the range it was fitted on; that is what "the trend was removed" means. Storing it in
float32 left the sum at **1e-6**, seven orders of magnitude off, and it now lands at float64
round-off. Nothing about the fit changed — only where the answer was written down.

**Thirteen is what the comparison saw, not how many numbers changed.** The float32 storage error
touches every value in those eight arrays, and the harness compares floats at `rtol=1e-6` — which
the error is comfortably inside (7.3e-06 absolute on values of order 100). It shows up in the
`sum` and nowhere else because the true sum is *zero*, so any absolute error there is an infinite
relative one. The recorded `min`, `max`, `std` and percentiles moved in their eighth digit and
were correctly judged unchanged.

**No phantom moves.** Not one value under any of the seven phantoms differs, because
`flatten_plane` hands `flatten_lines` float64 on every recorded chain. That is the same fact that
kept this defect out of the golden for the whole project.

**`_meta.python` moves 3.12.13 → 3.12.0**, the interpreter that regenerated the file — this
laptop's. It is recorded, never compared (it sits under `_meta`), and the numbers are comparable
because the golden was **verified stable on this machine before anything was edited**: same numpy
2.4.4, scipy 1.17.1 and scikit-image 0.26.0, zero drift.

### What the new block records, against what the old code did

`flatten_dtypes` levels one real phantom (`afm_tilted_polydisperse`, rescaled to 0–255) as five
dtypes. Running the pre-fix implementation on the same arrays:

| Input | Before | After | Pixels differing | Worst error |
|---|---|---|---|---|
| `uint8` | range `[0, 255]` | range `[-47.34, 131.87]` | **100 %** | **257** |
| `int32` | range `[-47, 131]` | range `[-47.34, 131.87]` | 100 % | 0.9999 |
| `bool` | `{0, 1}` | range `[-0.44, 1.01]` | 65 % | 1.44 |
| `float32` | same range | same range | 100 % | 7.3e-06 |
| `float64` | — | — | 0 % | 0 |

**The `uint8` row is the finding, and it is worse than the audit's.** The audit measured a ramp
whose residuals were all under 1 and reported "all zeros", which reads like a loss of resolution.
On an image with real structure the residuals are tens of nanometres and **negative ones wrap**:
`uint8(-47.34)` is 209. The levelled map came back with a bright band exactly where the sample sat
*below* its own row trend — every pit rendered as a peak, and nothing in the array to show for it.
The audit's own ramp is reproduced too: residual max `0.368`, recorded output all zeros.

`float32` differs in 100 % of pixels and by at most 7.3e-06 — the storage error, which is the
declared drift and nothing more.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| `dtype=np.float64`, hardcoded | Agrees with `flatten_plane` by coincidence rather than by rule, and demotes the one input `flatten_plane` keeps wide | The project only ever handled float64 |
| Promote integers and booleans, preserve float32 | Fixes the audit's case and leaves the two halves of flattening disagreeing on the dtype the SPM loader emits most | float32 output were a documented promise somewhere |
| `z = z.astype(float)` at the top | Copies the whole input to produce a result that is allocated anyway, and states the rule as a cast rather than as a promotion | The loop needed float rows for another reason |
| Raise on a non-floating input | An 8-bit image is what the SEM/TEM loader returns by construction; rejecting it would make a supported modality unlevellable. Typed rejection is M3-T13's, wholesale | Levelling were AFM-only |
| Leave it; document that callers must cast | Puts an invariant in prose that a one-line allocation can carry, and the callers are notebooks and a GUI that does not exist yet | The function were private |

## Compliance

- `tests/unit/test_flatten_dtype.py` — **17 tests**. An integer image keeps its residuals; a
  boolean one is not returned as a mask; both functions agree on dtype for all six dtypes tested;
  every dtype produces the residuals it computed, compared against a float64 reference that runs
  the same `polyfit`/`polyval` and stores them nowhere narrower; float64 is untouched; and
  levelling an 8-bit image leaves every row's slope at zero, which is the property the SEM/TEM
  path was silently not getting. **Restoring `np.empty_like(z)` turns 14 of the 17 red** — the
  three survivors are the float64 cases, which is the correct outcome: float64 never had the
  defect.
- Golden: `flatten_dtypes` is the permanent guard, the way D-03's
  `mean_abs_diff_vs_normalize_first` became one (ADR-0015). A regression to any narrower output
  dtype changes a recorded `dtype` string — on inputs the harness had none of before this task.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-13, and §5 R9 — *"Golden covers float; add an
  integer case"*
- `ADR-0025` — an absent number is not a substituted one; the same reasoning applied to scale
- **M3-T13** — the typed error taxonomy this ADR defers every rejection to
