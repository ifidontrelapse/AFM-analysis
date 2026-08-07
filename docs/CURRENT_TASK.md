# CURRENT TASK

**ID:** `M3-T25`
**Title:** Levelling can fit around a gap instead of refusing the scan
**Milestone:** M3 — Numerical correctness, twenty-third task
**Defect:** **B-060** (filed by M3-T13, whose rejection this completes) · **ADR:** **ADR-0036**
**Branch:** `sci/m3-numerical-correctness` (the consolidated branch — see the declared
deviation from PROJECT_RULES §7 in `STATE.md`)
**Status:** **done 2026-08-08.** Rewritten for the next task at the start of the next session;
the record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

Of the four findings open in M3, two are engineering (**B-060**, **B-064**) and two need
something this session cannot supply — **B6** blocks `M3-T16`, and **B-062** wants an operator's
view of a sensitivity trade-off. This is the first of the two.

M3-T13 made a non-finite value a rejection: *"a height map must be finite"*, checked at fourteen
entry points. That was the honest reading of what the code already did — `flatten_plane` had
always rejected NaN through `scipy.lstsq`. It was never the best behaviour available, and
ADR-0030 said so in its own text.

**A dropped scan line is a real artefact**, not malformed input. An AFM that loses feedback for
two lines produces a map with two rows of NaN and 4 094 perfectly good ones, and today the whole
scan is refused.

## What a masked fit recovers — measured

A 64×64 tilted scan with four particles and two dropped rows, levelled three ways and compared
against **levelling the same scan with no gap at all**:

| Strategy | plane coefficients `(a, b, c)` | max error on the intact pixels |
|---|---|---:|
| ungapped reference | `0.051149  0.031149  0.712928` | — |
| **masked fit** | `0.051186  0.031076  0.739255` | **0.0287 nm** |
| `nan_to_num(z, 0.0)` | `0.049587  0.031377  0.677039` | 0.1343 nm |

The masked fit recovers the plane to **0.03 nm**; zero-filling is **4.7× worse** and biases the
tilt itself — it does not merely add noise, it tells the fit that the sample dips to zero along
two lines.

Per row, for `flatten_lines`: a **partially** gapped row fits on its 54 finite points and keeps
the gap absent; a **fully** NaN row — which is exactly what a dropped scan line is — cannot be
fitted at all.

---

## The decisions this task has to make

### 1. Opt-in, not automatic

| | |
|---|---|
| **`allow_gaps=False` by default; the caller asks** ✅ | ADR-0030's contract holds everywhere it holds today, and the default path does not move a single number. A caller who knows their scan has gaps says so |
| Accept NaN automatically in levelling | Puts the library back where D-15 found it: levelling tolerates what detection refuses, and the two disagree about what an image is. That disagreement is the defect ADR-0030 closed six tasks ago |
| Relax `ensure_height_map` globally | Undoes ADR-0030 wholesale to serve one artefact |

**This does not make the pipeline gap-tolerant, and the ADR says so plainly.** The output of a
masked levelling still contains NaN, so `build_substrate_map` and both detectors still refuse it —
correctly, because nothing has decided what a substrate under a gap means. What the caller gains is
a levelled map they can crop, inspect or fill deliberately, instead of an exception. The rest is
filed, not implied.

### 2. The gap stays absent in the output

`NaN` in, `NaN` out, in exactly the same places. Not filled, not interpolated: an interpolated
value is a measurement nobody made, and this milestone has deleted seven substitute values
(ADR-0019, 0024, 0025, 0027, 0028, 0031, 0032). Interpolating a gap is a *feature* — with its own
decision about the method — not a consequence of this one.

### 3. A row that cannot be fitted comes back absent

A fully-NaN row has nothing to fit, and `polyfit` on fewer than `poly_order + 1` finite points is
not a fit either. Those rows return NaN, and the count is **warned about** — a scan that lost 40 %
of its lines should not level silently.

---

## Scope

**In scope**

1. `flatten_plane(z, *, allow_gaps=False)` — masked least squares over the finite pixels
2. `flatten_lines(z, poly_order=1, *, allow_gaps=False)` — per row, masked `polyfit`; rows with
   too few finite points come back absent, and the count is warned
3. `ensure_height_map(z, name, *, allow_gaps=False)` — the finiteness check becomes conditional,
   and nothing else about it does
4. Harness probes: a gapped phantom levelled both ways
5. Tests, including the comparison against the ungapped answer and against zero-filling

**Out of scope**

- **Gap-tolerant detection, substrate or measurement.** Filed as **B-065**; it needs a decision
  about what a substrate under a gap is, which is science, not plumbing
- **Interpolation.** A separate feature with its own method decision — **B-066**
- **B-064** — the constants. Next task, same session
- Detecting gaps automatically from a loader. `_read_nanoscope_z` produces no NaN today

---

## Expected blast radius, before measuring

- **Zero golden differences from behaviour.** `allow_gaps` defaults to `False`, so every existing
  call is byte-identical. The file changes only by the probes this task adds.
- If any recorded value moves, the default path was touched and the task is wrong.

---

## Definition of done

- [x] Both levelling functions take `allow_gaps`; the default path is untouched
- [x] A masked fit recovers the ungapped answer to **0.029 nm** and beats zero-filling **4.7×**
- [x] Rows that cannot be fitted are absent and counted, not silently zeroed
- [x] Tests — **12**, including that an intact map levels byte-identically either way
- [x] `make check` green — 464 tests; delta **5 differences, added keys only**, as predicted
- [x] ADR-0036; **B-065 and B-066 filed**; `Backlog.md` (B-060 → done), `STATE.md`, `Progress.md`,
      `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [x] Commit: `M3-T25: levelling can fit around a gap`

---

## What it turned up

**The advantage of the masked fit tracks the tilt**, which the synthetic scene could not show on
its own. Across the phantoms zero-filling costs **4.2×** on `afm_tilted_polydisperse` and only
1.2–1.7× on the flat ones — the fill corrupts the *plane*, so its damage is proportional to how
much plane there is to get wrong. A real AFM scan is tilted by construction, so the phantom that
matters is the one where the gap costs most.

**The prediction held exactly.** "Zero golden differences from behaviour; the file changes only by
the probes" — 5 differences, all `ADDED`. Worth recording because the previous task's prediction
was wrong by 70×, and the difference between the two is that this one changed a default-off flag
while that one changed a value every stage reads.

---

## Notes

ADR-0030 is not superseded. Its rule — a height map is finite — remains the default and remains
what every other entry point enforces. This adds a named exception that a caller has to ask for,
which is the difference between a contract with an escape hatch and a contract with a hole.
