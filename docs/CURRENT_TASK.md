# CURRENT TASK

**ID:** `M3-T13`
**Title:** A typed error taxonomy, and validation at every numerical entry point
**Milestone:** M3 — Numerical correctness, seventeenth task
**Defect:** **D-15** (medium) · **ADR:** **ADR-0030**
**Branch:** `sci/m3-numerical-correctness` (the consolidated branch — see the declared
deviation from PROJECT_RULES §7 in `STATE.md`)
**Status:** **done 2026-08-07.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

Two `medium` tasks are left and this is the one four other tasks have been deferring *to* by
name: M3-T06, M3-T07, M3-T08, M3-T17 and M3-T20 each declined to invent a rejection rule on the
grounds that D-15 owns all of them at once. A taxonomy of one is not a taxonomy, so the debt was
deliberately collected here.

## The defect, as the harness already records it

Eleven degenerate inputs × five entry points, today:

| Input | `flatten_plane` | `flatten_lines` | `build_substrate_map` | `detect_particles` | `estimate_log_threshold_adaptive` |
|---|---|---|---|---|---|
| `empty` (0×0) | ok | ok | ValueError | ValueError | ValueError |
| `single_pixel` | ok | LinAlgError | ValueError | **ok** | ok |
| `one_dimensional` | ValueError | IndexError | **TypeError** | **ok** | ValueError |
| `three_dimensional` | ValueError | ValueError | **RuntimeError** | **ok** | ok |
| `with_nan` | ValueError | **ok** | ValueError | **ok** | ok |
| `with_inf` | ValueError | **ok** | ValueError | **ok** | ok |
| constant / negative | ok | ok | ValueError · ok | ok | ok |

**Five different exception types, none of them ours, and `detect_particles` returns a clean empty
result for a 1-D array, a 3-D array, a NaN map and an infinite map.** That last row is the reason
this is not a cosmetics task: an unusable input and an empty sample are the same answer today.

The audit's own table says the same thing in prose: `AttributeError: 'str' object has no
attribute 'image'`, `too many values to unpack (expected 2)` from `flatten_plane`, `array must not
contain infs or NaNs` from `scipy.lstsq`, a Russian message from Otsu, and — for an all-zero array
— no error at all.

---

## The decisions this task has to make

### 1. What the taxonomy is

```
NanoscopeError(Exception)                    every error this library raises on purpose
├── InvalidInputError(…, ValueError)         the caller passed something no analysis can run on
│   ├── InvalidImageError                    the array: shape, dtype, emptiness, finiteness
│   └── InvalidParameterError                a scalar argument outside its domain
├── UnsupportedRequestError(…, ValueError)   the (modality, detector, mode) combination has no path
├── DataFormatError(…, ValueError)           a file or header we cannot read
├── MissingFileError(…, FileNotFoundError)   the file is not there
└── AnalysisFailedError(…, ValueError)       the input was valid; the analysis has no answer
```

**Every project error also inherits the builtin it replaces at that site.** `except ValueError`
in a notebook keeps working, which matters because the notebooks are the only callers this
library has. It is the `json.JSONDecodeError` pattern, and it makes the taxonomy adoptable in one
commit instead of a migration.

### 2. What a height map is

**2-D, numeric, non-empty, and finite.** The first three are structural. The fourth is a decision:

| | |
|---|---|
| Reject non-finite input at the entry ✅ | `flatten_plane` — step one of the documented chain — *already* rejects it, via `scipy.lstsq`. This makes the existing contract early, typed and identical everywhere, instead of a rule the first step happens to enforce and the second silently ignores |
| Accept it and let each function cope | Is today's behaviour: `flatten_lines` propagates NaN, `detect_particles` reports zero particles, `build_substrate_map` raises. Three answers to one question |
| Mask the non-finite values and fit around them | A real feature — a dropped scan line is a real thing — but it is a *scientific* change to what levelling computes, and this task is about rejections. File it |

**This supersedes part of ADR-0018 on exactly one input.** That ADR ruled that a non-positive or
`nan` maximum returns a default threshold rather than raising, because "zero particles is an
answer". That stays true for a *flat* or *negative* map, which is valid data with nothing in it.
A NaN map is not valid data, and the two cases were only ever conflated because the guard looked
at the maximum instead of at the input.

### 3. Where the check lives

One `ensure_height_map` in `core/validation.py`, called at each entry point, rather than a
hand-written check per function. The cost is a `np.isfinite(...).all()` pass — O(n), sub-millisecond
at 512×512 — and it will be measured, not assumed.

The harness records `raised_in`, which becomes the validator's name for every rejected input. No
attribution is lost: the golden's key already names the entry point that was called.

---

## Scope

**In scope**

1. `nanoscope/core/errors.py` — the taxonomy, no numpy import
2. `nanoscope/core/validation.py` — `ensure_height_map`, `ensure_positive`, and the one
   arity check the maths states (`flatten_lines` needs `poly_order + 1` columns)
3. Validation applied at the numerical entry points: `flatten_plane`, `flatten_lines`,
   `get_substrate_map`, `estimate_radius_otsu`, `estimate_rough_radius`, `build_substrate_map`,
   `estimate_log_threshold`, `estimate_log_threshold_adaptive`, `detect_particles`,
   `LogDetector.detect`, `YoloDetector.detect`, `measure_all_baseline`,
   `measure_geometry_from_mask`, and `run_pipeline`'s `data` argument
4. The nineteen existing deliberate `raise ValueError` sites re-typed to the taxonomy. They
   already name the parameter and its value; what they lack is a type a caller can catch

**Out of scope**

- **Masked / NaN-tolerant fitting.** Filed as **B-060**, because it changes what levelling
  *computes*, not what it rejects
- **B-059** (`nan <= 0` in `measure_all_baseline`) — a wrong number, not a missing rejection, and
  ADR-0010 keeps one defect to one commit
- The measurement schema (**M3-T14**) and the evaluation harness (**M3-T15**)
- Any new *capability* rule. `validate_request` already owns which combinations exist (M2-T10);
  it changes exception type here and nothing else

---

## Expected blast radius, before measuring

- **`degenerate_inputs` moves substantially**: error types become ours, `raised_in` becomes the
  validator, and — because ADR-0022 compares a message only when we wrote it — a batch of keys
  moves from `error_message_unchecked` back to `error_message`. Several `ok` cells become errors.
- **The seven phantoms must not move at all.** Every one of them is a valid image; if a recorded
  value changes, the validation is rejecting something real and the task is wrong.
- mypy: no expected movement. ruff: none.

---

## Definition of done

- [x] The taxonomy exists, every class documented with what raises it
- [x] Every entry point in the list validates its image argument; the existing raises carry a
      project type
- [x] `run_pipeline("not-data", cfg)` — the audit's first row — raises a typed error naming the
      argument and the type it got, before anything is constructed
- [x] Tests — **109**; the centre is 7 bad inputs × 10 entry points, 70 combinations and one
      error type, plus the same sweep proving a valid map passes all ten
- [x] `make check` green — 359 tests; delta **129 differences, no measured value**, and **no
      phantom value moved**
- [x] ADR-0030; **B-060 and B-061 filed**; `STATE.md`, `Progress.md`, `TASKS.md`,
      `PROJECT_CONTEXT.md`, ADR index, `Backlog.md`
- [x] Commit: `M3-T13: a typed error taxonomy, and validation at the entry`

---

## What it turned up

**The check that was too strict, caught by an older task's test.** Validating `radius_px` as
*positive* turned M3-T20's `test_and_that_costs_the_substrate_on_a_noisy_scan` red:
`estimate_rough_radius` returns **0** on an unscaled noisy scan, and `disk(0)` makes the opening
the identity. That is a defect — the "substrate" comes back equal to the image — but it is the
one **ADR-0025 measured and recorded**, so rejecting it here would have moved a number inside a
validation task. The check is non-negative and the question is filed as **B-061**. A regression
suite earning its keep in the direction that matters: stopping a change, not confirming one.

**ADR-0022's `_unchecked` category is now empty — 15 foreign messages to 0.** Not by deleting the
mechanism, which stays right, but because the entry points stopped handing out other projects'
sentences for inputs this project has an opinion about.

**Two existing tests had to change their subject**, and both say so in their docstrings rather
than being quietly rewritten: ADR-0018's NaN test and M3-T08's boolean test. A superseded rule is
worth more when the test that used to prove it explains what replaced it.

---

## Notes

The measure of this task is not how many checks it adds. It is that after it, the answer to
"what does this library do with input it cannot use" is one sentence instead of a table with five
exception types in it.
