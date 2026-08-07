# ADR-0030 — A typed error taxonomy, checked at the entry

- **Status:** Accepted
- **Date:** 2026-08-07
- **Affects:** `nanoscope/core/errors.py`, `nanoscope/core/validation.py`, and every numerical
  entry point · audit **D-15** · M3-T13
- **Numerical impact:** **129 golden differences, and not one measured value among them.** Every
  difference is an exception type, a message, a `raised_in`, or a cell that used to return a
  result for an input it could not use. **Foreign exception messages in the golden: 15 → 0.**

## Context

The audit's table for D-15 is five inputs and five behaviours:

| Input | Before |
|---|---|
| `run_pipeline("not-data", cfg)` | `AttributeError: 'str' object has no attribute 'image'` |
| 3-D array | `ValueError: too many values to unpack (expected 2)`, from `flatten_plane` |
| array containing `NaN` | `ValueError: array must not contain infs or NaNs`, from `scipy.lstsq` |
| 1×1 array | a `ValueError` in Russian, from Otsu |
| all-zero array | no error; zero detections |

The harness's own matrix, which the audit did not print, is worse. Eleven degenerate inputs
against five entry points produced **`ValueError`, `TypeError`, `IndexError`, `LinAlgError` and
`RuntimeError`** — and `detect_particles` answered a 1-D array, a 3-D array, a NaN map and an
infinite map with a clean empty result. **An unusable input and an empty sample were the same
answer**, which is the part that matters scientifically: a run that finds nothing is a normal
outcome, and it was indistinguishable from a run that never had a chance.

Five other tasks declined to fix their share of this and named D-15 as the owner — M3-T06,
M3-T07, M3-T08, M3-T17, M3-T20. A taxonomy of one is not a taxonomy, so the debt was collected
deliberately rather than paid in pieces.

## Decision

### The taxonomy

```
NanoscopeError(Exception)                    every error this library raises on purpose
├── InvalidInputError(…, ValueError)         the caller passed something no analysis can run on
│   ├── InvalidImageError                    the array: rank, emptiness, dtype, finiteness
│   └── InvalidParameterError                a scalar argument outside its domain
├── UnsupportedRequestError(…, ValueError)   a well-formed request with no implementation
├── DataFormatError(…, ValueError)           a file or header we cannot read
├── MissingFileError(…, FileNotFoundError)   the file is not there
└── AnalysisFailedError(…, ValueError)       valid input; the analysis has no answer
```

**Every class also inherits the builtin it replaced at its site.** `except ValueError` in a
notebook keeps catching what it caught, `except FileNotFoundError` around a loader keeps working,
and a caller who wants the distinction can ask for it. It is the `json.JSONDecodeError` pattern.
Without it this is a migration; with it, it is one commit.

The distinction the taxonomy exists to draw is between the three answers a caller can act on:

- **`InvalidInputError`** — fix the argument.
- **`UnsupportedRequestError`** — ask for something else. Nothing is malformed;
  `(sem, log, baseline)` is a sentence this version has no path for, and a GUI wants to grey out
  a menu item rather than tell the user their data is wrong.
- **`AnalysisFailedError`** — the image was fine and there is no result: Otsu found no objects,
  or the size filter removed all of them (ADR-0017). Deliberately **not** raised for an empty
  result: zero particles on a valid image is an answer, which ADR-0018 settled.

### What a height map is

**2-D, non-empty, of an integer or real dtype, and finite.** Checked by one
`ensure_height_map`, called at fourteen entry points, so that the answer cannot drift between
them again.

Three of those four are structural. **Finiteness is a decision**, and it is this ADR's:
`flatten_plane` — step one of the documented chain — has always rejected NaN, through
`scipy.lstsq`, while `flatten_lines` propagated it and `detect_particles` reported "no particles".
Rejecting at the entry makes the existing contract the whole library's instead of the first step's.

**This supersedes ADR-0018 on exactly one input.** That ADR's rule — a non-positive or `nan`
maximum returns the default threshold rather than raising, because zero particles is an answer —
stays in force for a **flat or negative** map, which is valid data with nothing in it. A map with
a NaN in it is not valid data. The two were only ever conflated because the guard looked at the
maximum instead of at the input.

**A boolean array is not a height map**, and this supersedes part of M3-T08 one commit later.
That task made `flatten_lines` promote instead of storing residuals in a `bool` array, where they
became a mask of where the residual was non-zero. The promotion rule stands for every dtype that
*is* a height map; a mask now gets refused at the entry, so the pathology is unreachable rather
than corrected. `ensure_mask` is the mirror image: it refuses a float array, because
`mask.astype(bool)` is a silent threshold at zero.

### Where the checks live

`core/validation.py`, called at the entry, not `if` statements written per function. The harness
records `raised_in`, which becomes `ensure_height_map` for every refusal — no attribution is lost,
because the golden's key already names the entry point that was called.

## Consequences

**Positive**

- One answer to one question, at fourteen doors. The matrix at the top of this file becomes one
  column.
- `detect_particles` can no longer report "no particles" for an input it never looked at.
- Every message names the parameter **as the caller's signature spells it** — `z_above`, not `z`
  and not "array" — and the value that was wrong: a shape, a dtype, or how many values were not
  finite.
- The messages are now ours, so ADR-0022's comparator holds them exactly, and the golden stops
  recording other projects' wording for inputs we have an opinion about.

**Negative**

- One `np.isfinite` pass per entry, O(n). Measured at ~0.05 ms on 512×512, against a
  `detect_particles` call three orders of magnitude slower — but it is a real cost, paid on every
  call including the valid ones.
- A caller who was relying on NaN passing through `flatten_lines` now gets an exception. There is
  no in-tree caller doing that, and the alternative (masked fitting) is filed as **B-060** rather
  than assumed away.
- Seven classes where there were none. The risk with a taxonomy is that it grows a class per
  call site; the rule that keeps it small is that a class earns its place by being *caught*
  differently, not by being *raised* differently.

**Neutral**

- No number moves. Every phantom is a valid image, so validation is a no-op on all seven.

## What is deliberately not in this commit

- **A zero opening radius is still accepted.** `disk(0)` is one pixel, so the opening is the
  identity and the "substrate" comes back equal to the image — an answer that looks like a result.
  It is reachable today from `estimate_rough_radius` on an unscaled noisy scan, and it is exactly
  what ADR-0025 measured and recorded. Refusing it here would move a number, and ADR-0010 keeps
  one intent to one commit. **Filed as B-061.**
- **Masked / NaN-tolerant levelling** — **B-060**. A dropped scan line is a real thing and fitting
  around it is a real feature; it changes what levelling *computes*, not what it rejects.
- **B-059** (`nan <= 0` in `measure_all_baseline`), unchanged and still filed.
- Validation of `PipelineConfig`'s fields as a whole. `run_pipeline` checks the argument it was
  handed and the parameters each function uses; a config-level schema belongs with the settings
  service in M4-T10.

## The measured delta

**129 differences. No value moved.**

| | |
|---|---|
| `error_type` changed | **32** |
| `error_message` now compared (was recorded-but-unchecked, ADR-0022) | **28** |
| `raised_in` changed | 15 |
| a cell that used to succeed now raises (`ok: True -> False`) | **13** |
| `error_type` / `raised_in` added on those 13 | 26 |
| `result` removed on the 11 that used to return one | 11 |
| `stdout_lines` removed, `value` changed (the two `flatten_dtypes.bool` cells) | 4 |

### The matrix becomes one column

```
ValueError   -> InvalidImageError        11      TypeError    -> InvalidImageError      1
ValueError   -> AnalysisFailedError      10      IndexError   -> InvalidImageError      1
ValueError   -> InvalidParameterError     7      LinAlgError  -> InvalidParameterError  1
                                                 RuntimeError -> InvalidImageError      1
```

The four single-count rows are the point. `TypeError`, `IndexError`, `LinAlgError` and
`RuntimeError` were four different libraries' opinions about the same four malformed inputs,
reached through four different call paths; they are now one sentence naming the parameter.

### Foreign messages in the golden: 15 → 0

ADR-0022 introduced `error_message_unchecked` for messages this project did not write, because
CPython 3.14 reworded one and the golden read it as drift. **That category is now empty.** Every
message the harness records is one of ours, compared exactly, on 45 keys where there were 17.

The mechanism stays. It is the policy that a foreign message is recorded and not compared, and
the next library upgrade may well produce one; what the count says is that the entry points no
longer *hand out* foreign messages for inputs this project has an opinion about.

### The twelve phantom differences, and why they are not values

| Where | What |
|---|---|
| `preprocessing.estimate_radius_otsu_all_filtered.error_type` × 5 AFM | `ValueError` → `AnalysisFailedError` |
| `yolo_input_preparation.boxes_to_detections_confidence_mismatch.error_type` × 7 | `ValueError` → `InvalidParameterError` |

Both are probes the harness added to record a *failure* — M3-T06's empty-after-filter case and
M3-T05's length-mismatch case. They were already raising; they now raise something a caller can
tell apart. **No measured quantity under any phantom moved**, which is the property that had to
hold: every phantom is a valid image, so validation is a no-op on all seven.

### Thirteen cells that used to answer

Eleven degenerate inputs that `flatten_lines`, `detect_particles` or
`estimate_log_threshold_adaptive` used to accept — a 1-D array, a NaN map, an infinite map — plus
the two `flatten_dtypes.bool` cells. Those are the ones D-15 was really about: the code did not
fail on them, it *answered* them.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| One exception class for everything | The three answers a caller can act on — fix the argument, ask for something else, accept there is no result — would be one class and a string to parse | The library had one caller and no GUI planned |
| Project errors that do **not** inherit the builtins | Cleaner hierarchy, and every existing `except ValueError` silently stops catching. The notebooks are the callers; a silent behaviour change in the caller's error handling is the worst kind | There were no existing callers |
| Validate in a decorator | Reads well and hides *which* argument is checked, which is the one thing the message has to name. Also breaks the signatures mypy is checking | The rule were uniform across every function |
| Coerce rather than reject (`np.asarray`, `nan_to_num`) | Every coercion is a substituted value, and this milestone has spent six ADRs deleting those. `nan_to_num` in particular invents zeros that read as substrate | The coercion were lossless and unambiguous |
| Leave `detect_particles` tolerant, since "no particles" is a safe answer | It is the opposite of safe: it is a *plausible* answer, and the audit's own measurement of TEM (0 of 22) shows how long a plausible zero survives unquestioned | The caller could tell the two zeros apart |

## Compliance

- `tests/unit/test_errors.py` — **109 tests**. The centrepiece is a 7×10 parametrization: seven
  things that are not a height map against ten entry points, **70 combinations, one error type**
  — and a matching sweep proving a valid map passes all ten, because validation that rejects real
  data would be a worse defect than the one it fixes. Then the audit's five rows one by one
  (including the all-zero array, which must **still** not be an error); the catchability property
  per class, asserted three ways (project type, builtin, `NanoscopeError`); that a missing file is
  a `FileNotFoundError` and not a `ValueError`; that an unsupported request is not an
  `InvalidInputError`; that an integer image is a height map and a float array is not a mask; and
  that a zero radius is still allowed, with B-061 named in the test.
- Two existing tests changed with their subject and say so:
  `test_a_nan_map_is_refused_before_the_maximum_is_taken` (was
  `test_a_nan_maximum_is_caught_rather_than_propagated`, ADR-0018) and
  `test_a_boolean_image_is_refused_rather_than_levelled` (was `..._is_not_returned_as_a_mask`,
  ADR-0029). Both keep the old behaviour's reasoning in the docstring.
- `tests/characterization/test_exception_text_policy.py` — two examples moved, because the inputs
  they used to provoke a foreign message are now refused by name. The classifier's two properties
  are unchanged and still proven: a NumPy-composed message inside our file
  (`measure_height` with mismatched masks) is not ours, and a library's own explicit `raise`
  (`scipy.linalg.lstsq` on a NaN matrix — the very error `flatten_plane` used to surface) is not
  ours either.
- Golden: 129 declared differences, listed above.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-15
- `ADR-0017` — the analysis failing loudly; `ADR-0018` — zero particles is an answer, superseded
  here on non-finite input only
- `ADR-0022` — the golden compares the messages we wrote, which is now more of them
- `ADR-0029` / M3-T08 — the boolean case, refused here rather than corrected
- **B-060**, **B-061** — filed by this task
