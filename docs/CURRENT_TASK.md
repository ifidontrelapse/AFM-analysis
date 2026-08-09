# CURRENT TASK

**ID:** `M3-T19`
**Title:** The annotations that lie about `None` — the last unblocked engineering item in M3
**Milestone:** M3 — Numerical correctness, twenty-fifth task
**Defect:** **M3-T19** (found by mypy in M1-T04) · **ADR:** none — see "No ADR, and why"
**Branch:** `sci/m3-numerical-correctness` (the consolidated branch — see the declared
deviation from PROJECT_RULES §7 in `STATE.md`)
**Status:** **done 2026-08-09.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

It is the only thing left in M3 that needs no decision. `M3-T16` waits on **B6**; **B-062**,
**B-065**, **B-066** and **B-067** each need an operator's view or a new algorithm, and none of
them is an afternoon.

M3-T19 as filed is three mypy errors in `log.py`:

```python
responses = []                      # list[float]
...
responses = np.array(responses)     # rebound to ndarray  → error [assignment]
responses.min(), responses.max()    # → 2 × error [attr-defined]
```

Reading the file for it turned up the same *kind* of fault two functions down, and again one file
over:

```python
def detect_particles(..., threshold: float = None, ...)          # log.py:192
def build_substrate_map(..., manual_radius_px: float = None, ...) # substrate.py:301
```

Both bodies already branch on `None` (`log.py:237`, `substrate.py:334`) — the *documented* meaning
of the default is "not supplied, compute it". The annotation says the opposite: that the parameter
is always a number. Same defect as the rebinding, stated the other way round — **an annotation
that does not describe the value the code actually carries** — and mypy reports 6 of its 12 errors
against those two patterns.

---

## Scope

**In scope**

1. `log.py` — stop rebinding: the accumulator keeps its list type, the array gets its own name
2. `log.py:192` — `threshold: float | None = None`, which also removes the caller's error at
   `log.py:381` (`LogDetector.threshold` is already `float | None`)
3. `substrate.py:301` — `manual_radius_px: float | None = None`
4. Two tests: an **explicit** `None` is accepted at both entry points and gives the same answer as
   omitting the argument — the meaning the annotation now states

**Out of scope**

- The remaining 6 mypy errors. Four are in `pipeline.py`: three pass `ndarray | None` into
  functions that require an array — a real question about what detect mode returns — and one is
  the `if/elif` detector dispatch the port exists to remove. Both are M4's, not a rename.
  `yolo.py:124` and `plots.py:37` are third-party overloads
- `r = max(int(sigma), 1)` at `log.py:165`. It is a truncation in the family M3-T24 hunted, but it
  sizes a *neighbourhood window for a peak lookup*, not a physical radius, and changing it moves
  numbers. Not smuggled in (ADR-0010); recorded here and not filed, because a window that is one
  pixel small on a blob's peak is not a defect anyone can demonstrate

---

## No ADR, and why

M3's gate (ADR-0010) is "one defect, one commit, one ADR, one golden update" and it exists because
these tasks *move numbers*. This one cannot: an annotation is not executed. **No decision is
made** — the annotations are being made to agree with the branches already in the code, and where
the code and the annotation disagreed, the code was right. **M3-T18 set this precedent** and is
recorded as such in `TASKS.md`.

What this task still owes the gate is the *measurement*: the golden delta, stated and verified,
and the mypy count before and after.

---

## Expected blast radius, before measuring

- **Zero golden differences**, and unlike previous tasks that predicted zero, here it is a
  property of the change rather than an expectation: no executable line changes.
- **mypy 12 → 6.** If it lands anywhere else, something other than an annotation moved.

---

## Definition of done

- [x] No rebinding in `estimate_log_threshold_adaptive`
- [x] Both defaults annotated `float | None`
- [x] Tests — explicit `None` at both entry points, equal to the omitted argument
- [x] `make check` green; delta **zero, golden byte-identical**
- [x] `make types` reports **6**
- [x] `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M3-T19: the annotations stop lying about None`

---

## What it turned up

**The filed defect was one instance of a class.** M3-T19 was written down as three mypy errors
about a rebinding; the same fault — an annotation that does not describe the value the code
carries — was sitting in the signature of the same function and in `build_substrate_map`, stated
the other way round as an implicit Optional. Fixing what was filed would have left half of it.

**Zero delta, for once by construction.** Every previous M3 task predicted a golden delta and
measured it. This one could not move a number: no executable line changed. The measurement that
matters here is mypy's count, **12 → 6**, and the six that remain are not annotation drift — four
are `pipeline.py`'s detect-mode `ndarray | None` and the `if/elif` detector dispatch, which are
M4's questions, and two are third-party overloads.

**M3's engineering queue is now empty except `M3-T16`, which is blocked on B6.** All five exit
criteria are met; the Roadmap's three stale checkboxes (M3-T13, T14, T15, all closed 2026-08-07)
were ticked in the same commit. What is left in the milestone is four findings that each need an
operator's decision, not an afternoon: **B-062**, **B-065**, **B-066**, **B-067**.
