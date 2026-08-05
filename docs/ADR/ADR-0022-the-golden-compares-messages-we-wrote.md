# ADR-0022 — The golden compares the messages we wrote, and only those

- **Status:** Accepted
- **Date:** 2026-08-05
- **Affects:** `tests/characterization/capture.py` · backlog **B-058** · decision from the
  operator 2026-08-05
- **Numerical impact:** none. **15 keys are renamed** (`error_message` →
  `error_message_unchecked`); no value changes, and no value stops being recorded.

## Context

`capture.py` records the exception a degenerate input provokes — its type, its message, and the
function it came out of — and `compare` checked all three exactly. Most of those sentences were
never ours:

```
too many values to unpack (expected 2)              CPython
tuple index out of range                            CPython
Only 2-D and 3-D images supported.                  scikit-image
array must not contain infs or NaNs                 numpy
Otsu found 804 objects, none with a radius of ...   ours
```

CPython 3.14 reworded the first one to `(expected 2, got 3)`. The first real CI run resolved
3.14 — `requires-python = ">=3.12"` — and reported it as **characterization drift**: a red gate,
with no scientific change behind it (M1-T08). CI was pinned to 3.12 as the immediate fix, and the
fragility was filed as **B-058**, explicitly needing an ADR rather than a quiet edit, because
changing what the harness records is a change to the refactor contract (PROJECT_RULES §4.3,
ADR-0008).

The golden's job is to notice when *our* behaviour moves. A sentence composed by the interpreter
is not our behaviour; that a `ValueError` comes out of `flatten_plane` when handed a 1-D array
absolutely is.

## Decision

**The exception type and the raising function are always compared. The message is compared only
when this project wrote it; every other message is recorded under a key `compare` skips.**

```python
def _we_wrote_this_message(frame) -> bool:
    return "nanoscope" in Path(frame.filename).parts and "raise " in (frame.line or "")
```

**Both signals, because either alone is wrong.**

- *Filename alone* would claim `h, w = z.shape` in our own `flatten_plane` — the exact M1-T08
  case, where the file is ours and the wording is CPython's.
- *`raise` alone* would claim `raise TypeError('Only 2-D and 3-D images supported.')`, which is
  scikit-image's sentence, in scikit-image's file.

The two together select exactly the messages a developer here typed, which are the ones
PROJECT_RULES §3 governs — the ones that must name the parameter and its value, and whose wording
is therefore a real contract. After the change **7 messages remain compared, all of them
`estimate_radius_otsu`'s**, and 15 move to `_unchecked`.

**Recorded, not dropped.** `error_message_unchecked` still appears in the golden and still gets
regenerated. A reader diagnosing a failure needs the text; the harness simply stops promising
that somebody else's wording is stable. Any key ending in `_unchecked` is skipped by `compare`,
which makes the rule reusable the next time something worth recording is not worth promising.

## Consequences

**Positive**

- **A Python upgrade no longer reads as drift.** This was the stated blocker on touching the
  interpreter version, and `docs/STATE.md` listed it as a precondition for any upgrade.
- The contract gets *sharper*, not looser: what actually failed and where is still compared
  exactly, and the messages we are responsible for are still compared character by character.
- One suffix convention, one line in `compare`.

**Negative**

- If a library changes its exception *type* we still catch it, but if it changes only its wording
  in a way that indicates a real behavioural change, we no longer see it. That is the trade being
  made, and the type plus the raising function are the compensating signals.
- `_we_wrote_this_message` is a heuristic over a traceback frame. A `raise` inside a
  multi-statement line, or a vendored copy of our code under a different path, could fool it.
  Three tests pin the three cases that matter today.

**Neutral**

- 15 keys renamed, no values changed. The golden is the same size and records the same text.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Normalise known rewordings (regex the message before comparing) | A list of every CPython and library wording, maintained forever, that grows on every upgrade. It also hides a real change behind a rule written for a cosmetic one | Only one or two messages were affected |
| Compare `error_type` only, drop messages entirely | Throws away our own contract — `min_size_pixel=5 px (the largest is 3.48 px)` is behaviour, and M3-T06 made it so deliberately | Our exceptions carried no information |
| Record messages but exclude all of them from comparison | Simpler rule, and the one the backlog suggested. It gives up the messages we do control, including the ones a future task will change on purpose | We never wrote informative exceptions |
| Pin the interpreter forever | Already the status quo, and it converts a test-harness weakness into a permanent constraint on the runtime | Python never changed a message |

## Compliance

- `tests/characterization/test_exception_text_policy.py` — 6 tests: our own `raise` is ours; a
  CPython message raised in *our* file is not; a library's explicit `raise` is not (so neither
  signal is redundant); `compare` ignores a reworded `_unchecked` value, reports a reworded one of
  ours, and still reports a changed `error_type` either way.
- Golden: regenerated, 15 keys renamed, 0 values changed.

## References

- `docs/Backlog.md` **B-058**, which specified that this needed an ADR
- `ADR-0008` — the golden as the refactor contract, and why what it records is not edited quietly
- M1-T08 — the CI run where a Python minor version read as a scientific change
