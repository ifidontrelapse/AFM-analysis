# ADR-0010 — One defect, one commit, one ADR, one golden update

- **Status:** Accepted
- **Date:** 2026-08-03
- **Affects:** all of M3 · every future change to numerical behaviour

## Context

The Phase 0 audit confirmed 24 defects. Four of them will change scientific output when
fixed, by design:

| Defect | Current behaviour | Measured impact |
|---|---|---|
| **D-03** | YOLO input casts to `uint8` before normalising | 12.6% of dynamic range reaches the model; 19–47 grey levels out of 256 |
| **D-04** | `int(min_size_nm / pixel_size_nm)` floors to 0 | Noise filter disabled on **108 of 120** of the operator's scans |
| **D-10** | Half-integer opening radii | Even-sized structuring element, no centre pixel, half-pixel shift in `z_result` |
| **D-12** | Detector keeps the bright side of the Otsu threshold | TEM finds **0 of 22** particles; it characterises the background |

Fixing any of these changes detections, radii, and every measurement downstream. If two
are fixed in one commit, or a fix rides along with a refactor, the deltas become
inseparable and no one can say afterwards which change moved which number — or whether it
moved in the right direction.

This is not hypothetical. D-04 corrupts the radius statistics that set the LoG sigma
range, so its blast radius covers every detection on the majority of real data.

## Decision

**Every change to numerical behaviour is isolated.** A commit that moves a golden number
contains exactly one intent and all four of:

1. **An ADR** stating what was wrong, what the correct behaviour is, and why.
2. **A test** that fails before and passes after.
3. **The regenerated golden file**, in the same commit.
4. **A `Progress.md` entry** quantifying the delta — per phantom, per quantity.

And obeys these rules:

- **Never bundle a numerical fix with a refactor.** M2 (structure) completes before M3
  (numerics) begins on the same module.
- **Never bundle two numerical fixes.** One defect, one commit.
- **Branch prefix `sci/`** marks a branch that intentionally changes scientific output.
- **Defects whose correct behaviour is a scientific question require operator sign-off
  before the fix is written** — not after. D-04 (what does minimum particle size mean at
  9.77 nm/px?), D-10 (which rounding rule?) and D-12 (explicit polarity or auto-detection?)
  are blocked on this and marked so in `STATE.md`.
- Fix order follows blast radius, largest last: D-01 → D-04 → D-03 → D-21 → the rest.

## Consequences

**Positive**

- Every number that moves has exactly one named cause and a recorded magnitude.
- A fix that turns out to be wrong is revertable in one commit without unpicking a refactor.
- The operator can review scientific changes one at a time, in their own language, instead
  of auditing a 40-file diff.
- Results produced before and after a given commit can be compared meaningfully, because
  the changelog says what changed.

**Negative**

- Slow. Sixteen M3 tasks, each with its own branch, ADR, review and re-baseline, where a
  single "fix the numerics" pass would take a fraction of the time.
- Sequential dependencies: D-04 changes the radius statistics that D-03's characterisation
  is measured against, so goldens are re-baselined repeatedly and each re-baseline must be
  read carefully.
- Three tasks are blocked on operator decisions and will stall if answers are slow.
- More ADRs than some readers will want.

**Neutral**

- The audit's risk register (§5, R1–R13) already assigns each change a probability of
  moving output; that table becomes the M3 sequencing plan.

## Alternatives considered

| Alternative | Why not |
|---|---|
| One "fix all numerics" pass | Produces a single unattributable delta across every measurement. Nobody could verify it, and a partial revert would be impossible. |
| Fix numerics during the M2 move | Cheaper in keystrokes, and it would destroy the only guarantee M2 has — that zero drift means the move was correct. |
| Fix only the critical defects, live with the rest | The mediums are the ones that produce plausible-looking wrong answers (D-06 over-counts, D-08 crashes on empty results, D-17 ships four schemas). Plausible-looking wrong answers are worse than crashes. |
| Fix without operator sign-off, using engineering judgment | D-04, D-10 and D-12 are questions about what a particle *is* and how a surface should be estimated. That is the operator's domain, and guessing would silently redefine the science. |

## Compliance

- CI fails on golden drift; a golden update without an ADR reference in the commit body
  fails review.
- Each M3 task maps to exactly one defect ID from the audit.
- `Progress.md` contains a quantified delta for every commit that moved a number.
- `sci/` branches are never squashed together.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §2 (defect register), §5 (risk register)
- `docs/audit/characterization-baseline.md` §3
- `docs/TASKS.md` M3
- ADR-0008
