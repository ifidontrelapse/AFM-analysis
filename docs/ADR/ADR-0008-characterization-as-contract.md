# ADR-0008 — The characterization golden file is the refactor contract

- **Status:** Accepted
- **Date:** 2026-08-03
- **Affects:** `tests/characterization/`, all of M2 and M3

## Context

M2 moves every scientific module in the repository into a new package structure, and M3
then changes numerical behaviour deliberately. Both are dangerous in the same way: the
science has no unit tests. The only existing test file has no assertions, catches the
wrong exception, and references a path absent from a clean checkout (D-20).

What does exist is a Phase 0 characterization harness: 8 deterministic seeded phantoms
with ground truth, a capture/compare runner, and a committed golden file recording the
current behaviour of every numerical stage at `rtol=1e-6, atol=1e-9` — including the
exception types and messages produced by degenerate inputs. It runs in ~100 s, CPU only,
no weights, no network.

Some of the recorded numbers are scientifically **wrong** — the audit says which and why.
That is the point: it records what the code *does*, not what it should do.

## Decision

The golden file is the **contract for structural change**.

1. **A refactor must leave every golden number unchanged.** `python
   tests/characterization/capture.py` reporting zero drift is a merge requirement for
   every M2 task.
2. **If a number moves, one of two things is true.** Either the refactor has a bug — fix
   it — or the change was intentional, in which case the commit must contain **all four**:
   an ADR, a test proving the new behaviour, the regenerated golden, and a `Progress.md`
   entry quantifying the delta.
3. **`--write` without an ADR is a rule violation.** Re-baselining is a declaration, not a
   convenience.
4. The harness is wired into `pytest` (M1-T05) so it runs by default rather than by
   discipline.
5. Dependency upgrades that move numbers (scikit-image, SciPy) are re-baselined in their
   own commit together with the version bump, with the delta recorded.
6. The phantom set is not throwaway audit scaffolding — it becomes the fixture base for
   the M3-T15 evaluation harness, which scores detection against the ground truth the
   phantoms already carry.

## Consequences

**Positive**

- A large refactor becomes an engineering task with a pass/fail signal instead of a gamble.
- Intentional scientific changes become visible, attributable and reviewable — the delta
  is printed, per quantity, per phantom.
- Degenerate-input behaviour is pinned, so the M3-T13 error-taxonomy work has a documented
  "before".
- No binary test data enters git: phantoms are generated in-process from seeds.

**Negative**

- ~100 s per run is slow enough to discourage running it often; it needs a `slow` marker
  and a fast subset for inner loops.
- A tolerance of `rtol=1e-6` will produce occasional false positives from legitimate
  floating-point reassociation (e.g. changing operation order), which must be
  investigated rather than waved through.
- **Segmentation is not covered.** SAM2 inference is not reproducible enough to golden, so
  `segmentation.py` is the least protected module in the repository — which is why it
  moves last in M2.
- The golden encodes current *wrong* behaviour; a reader who mistakes it for a
  specification will draw false conclusions. The document says so explicitly, in bold.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Write proper unit tests first, then refactor | Correct in principle, months of work in practice, and it requires deciding what is correct — which is exactly the M3 question that needs the operator. Characterization lets structure proceed while correctness is still being decided. |
| Refactor carefully, review thoroughly, no golden | 2 000 lines of numerical code with five import cycles. "Carefully" is not a control. |
| Loose tolerance (`rtol=1e-3`) | Would hide real defects introduced by a move. The tight tolerance catches accidents; scientific significance is judged separately by a human. |
| Golden on real scan data | `data/` is gitignored and belongs to the operator (B6). Phantoms are deterministic, small, and carry ground truth that real data does not. |

## Compliance

- Every M2 pull request includes the capture output showing zero drift.
- Every golden update commit contains an ADR reference in the commit body.
- CI runs the comparison; a drift exits non-zero and fails the build.
- The `afm_coarse_pixels` phantom (9.77 nm/px, the real-data median) must remain in the
  set — it is the only fixture pinning defect D-04.

## References

- `docs/audit/characterization-baseline.md`
- `tests/characterization/{phantoms.py,capture.py,golden/baseline.json}`
- ADR-0010
