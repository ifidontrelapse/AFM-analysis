# CURRENT TASK

**ID:** `M3-T08`
**Title:** `flatten_lines` promotes the way `flatten_plane` does
**Milestone:** M3 — Numerical correctness, sixteenth task
**Defect:** **D-13** (medium) · **ADR:** **ADR-0029**
**Branch:** `sci/m3-numerical-correctness` (the consolidated branch — see the declared
deviation from PROJECT_RULES §7 in `STATE.md`)
**Status:** **done 2026-08-07.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

Every `critical` and every `high` defect is closed. `STATE.md` names the remaining `medium`
ones in order — **T08, T13, T14** — and T08 is the only one of the three that is a *number*
rather than a contract: T13 designs an error taxonomy and T14 unifies a schema, both of which
touch every entry point. This one is one line in one function.

It is also the task this laptop can finish. There are no weights and no `data/` here, so
anything that has to execute YOLO or SAM2, or re-read the 628 local scans, cannot be verified;
`flatten_lines` is pure NumPy/SciPy and the whole gate runs on the CI-shaped environment.

---

## The defect

`nanoscope/core/science/preprocessing/flatten.py:46`

```python
result = np.empty_like(z)          # <- keeps the input's dtype
for i, row in enumerate(z):
    coeffs = np.polyfit(xi, row, poly_order)
    result[i] = row - np.polyval(coeffs, xi)   # <- float64 residuals, truncated on assignment
```

`np.polyfit` and `np.polyval` compute in float64 whatever they are given, so the residual is
fractional by construction — a levelled row is *mostly* fractional, since the trend it subtracts
is the row's own fit. Assigning it into an integer array rounds every value toward zero.

Measured here on a `uint8` ramp: residual max **0.368**, recorded output **all zeros**. The audit
measured its own ramp and got the same shape of answer (correct 0.5625, actual all zeros).

**`flatten_plane` does not have the defect** — it returns `z - plane` with `plane` float64, so
NumPy promotes for it. The two halves of "flattening" therefore disagree about dtype, which is
what makes this a defect rather than a preference: `flatten_plane(z).dtype` is `float64` for a
`uint8` input and `flatten_lines(z).dtype` is `uint8`.

**Boolean input is worse than truncation, and the audit did not measure it.** `result[i] = row -
polyval(...)` into a `bool` array stores `residual != 0`, so the "levelled topography" comes back
as a mask of where the residual was non-zero — max **1.0** where the real residual max is 0.45.

### Who reaches it

Not the documented chain: `flatten_lines(flatten_plane(z))` already receives float64, which is
why the golden's five AFM phantoms have never seen this. Live callers are the ones that hand
`flatten_lines` an array directly:

- **`load_microscopy_image` returns `uint8`** — `cv2.imread(..., IMREAD_GRAYSCALE)` — and it is
  the only file entry point for SEM/TEM. Levelling one of those images is a `flatten_lines` call
  on integer data.
- **`load_afm(fmt="npy")`** passes through whatever the `.npy` holds, integers included.
- `README.md`, `project.md` and `PROJECT_CONTEXT.md` all document `flatten_lines` as a public
  function callable on its own.

---

## The decision this task has to make

The task title in `TASKS.md` says "must promote dtype like `flatten_plane` does", and that is a
rule, not a hint. `flatten_plane`'s rule is NumPy's own: the input dtype combined with float64.

| Option | |
|---|---|
| `np.promote_types(z.dtype, np.float64)` ✅ | *Is* `flatten_plane`'s rule, written out. Integers and bools become float64; float64 stays; a wider float stays wide |
| `dtype=np.float64`, unconditionally | Identical for every dtype `np.polyfit` accepts today, and *demotes* a `longdouble` input, which `flatten_plane` preserves. Two rules that agree by coincidence are two rules |
| Cast the input — `z = z.astype(float)` first | Copies the array to compute a result that was going to be allocated anyway |
| Raise on non-float input | An 8-bit image is not a malformed input; it is what the SEM/TEM loader returns by construction. And typed input validation is **M3-T13**, wholesale |

**float32 in becomes float64 out**, and that is declared drift rather than an accident: it is
what `flatten_plane` already does with a float32 input, and what NumPy would do if the expression
were written `z - trend` instead of assigned into a pre-allocated buffer.

---

## Scope

**In scope**

1. The one line in `flatten_lines`, plus a docstring that states the returned dtype
2. A harness block that records levelling across dtypes — the integer case the audit's own
   remediation note (R9) asked for: *"Golden covers float; add an integer case"*
3. Unit tests, including the dtype-agreement invariant between the two functions

**Out of scope**

- **Typed validation** of `z` — non-numeric dtypes, 1-D, 3-D, NaN. `np.promote_types` raises its
  own `TypeError` on a string array one line earlier than `np.polyfit` did; making that a project
  error is **M3-T13**, which owns every numerical entry point at once
- `flatten_plane`. It is correct here and nothing in it moves
- **B-059** (`nan <= 0` in `measure_all_baseline`), a different defect in a different file, which
  gets its own commit (ADR-0010)

---

## Expected blast radius, before measuring

- **The five AFM phantoms and both image phantoms: no change.** Every recorded `flatten_lines`
  call in a phantom chain is fed `flatten_plane`'s float64 output.
- **`degenerate_inputs`: dtype changes.** All twelve are float32, so the entries that succeed
  today record `dtype: float32` and will record `float64`, with the value statistics moving in
  the last bits of float32 precision. That is the drift this ADR declares.
- **New keys** for the dtype block.
- mypy and ruff: no expected movement.

---

## Definition of done

- [x] `flatten_lines` allocates with `np.promote_types(z.dtype, np.float64)`; the docstring says
      so and says why
- [x] Harness records levelling for `uint8` / `int32` / `bool` / `float32` / `float64` inputs of
      one real phantom image, both functions, dtype included
- [x] Tests — 17; restoring `np.empty_like(z)` turns **14** red, the three survivors being the
      float64 cases, which never had the defect
- [x] `make check` green — 249 tests; delta: **13 differences — 8 dtypes, 4 sums, 1 added group**;
      mypy unchanged at 12
- [x] ADR-0029; `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [x] Commit: `M3-T08: flatten_lines promotes the way flatten_plane does`

---

## What it turned up

**The audit understated the defect in two directions.** It measured a `uint8` ramp whose
residuals were all under 1, got "all zeros", and filed it as truncation. On an image with real
structure the residuals are tens of nanometres and **an integer output wraps the negative ones** —
`uint8(-47.34)` is 209. On the newly recorded 8-bit phantom **100 % of pixels are wrong, by up to
257**, and every pit came back rendered as a peak. That is not a degraded map; it is features that
are not there. And **boolean input was never measured at all**: levelling a mask returned a mask
of where the residual was non-zero.

**The four moved sums are the fix as a physical property.** A least-squares residual sums to zero
over the range it was fitted on — that is what "the trend was removed" means. Storing it in
float32 left the sum at **1e-6**; it now lands at float64 round-off. The fit never changed, only
where the answer was written down.

**mypy is unchanged at 12, and that is the second time this milestone.** A dtype that is correct
for one input and wrong for another has no static shadow, exactly as M3-T02's unit error had none.
Between them they mark the class of defect the type checker cannot be the guard for.

---

## Notes

The claim this task can and cannot make: it fixes what levelling *returns* for an integer image.
Whether levelling an 8-bit SEM image is the right preprocessing step at all is a different
question, and nothing in the gate answers it — **M3-T15** again.
