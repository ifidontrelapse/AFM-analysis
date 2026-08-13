# ADR-0074 — A ruler is not an annotation, and the word "measurement" is taken

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** operator + agent (M7-T05)
- **Affects:** `core/science/metrology`, `core/entities`, `infrastructure/storage`, `gui` · M7

## Context

Four tools in, every shape this project stores describes *a thing*. A ruler describes **a distance
between two things**, and it is the first output in the project that no algorithm produced — which
the roadmap's risk line for M7 names: *"manual measurements are a new output and get their own
tests."*

## Decision

### 1. It is not an annotation, for the reason that refused the point

A line has no area. ADR-0044 stores shapes with extent and refuses a zero-area one twice, so a ruler
would fail both checks; forcing one through as a degenerate box is the invention ADR-0071 declined
one task earlier. It gets a table.

### 2. The word "measurement" is taken

`measurements.csv` is what an analysis run produces (ADR-0031, ADR-0042) — derived, re-runnable,
shaped by its producer. What an operator drew by hand is none of those, and calling both
"measurements" makes *"where are the measurements?"* a question with two answers. The table is
**`rulers`**.

### 3. One table, two tools

A profile line (M7-T06) is the same geometry read differently, so the row carries a `kind` and the
migration (schema **v8**) happens once.

### 4. The length is computed, never stored

`distance_px` over the endpoints, every time. A stored length is a second answer waiting to disagree
with the points it came from — the same reason ADR-0072 derives a polygon's box rather than accepting
one.

The arithmetic lives in `core/science/metrology.py`: two points and Pythagoras is still science, and
in a widget it would be the first science in `gui/` in seven milestones. It is a **new output**, so
it ships with its own tests.

### 5. Without a scale there is no length in nanometres

Pixels always; nanometres only when the project recorded a scale, and the words *"scale unknown"*
otherwise. This is ADR-0025's rule at the first surface that **produces** a physical number rather
than reading one — and a non-positive scale is a *wrong* answer rather than an absent one, so it
raises.

### 6. It goes through the command stack, like every tool since M7-T02

## Consequences

**Positive** — an operator can measure what the detector did not; the number cannot drift from the
line it came from; the profile tool inherits a table and a migration it does not have to write.

**Negative** — a ruler has no units control: an operator working in µm reads nanometres and divides.
Adding one is a display preference (ADR-0047's scope rule) and nothing has asked for it.

**Neutral** — rulers are drawn under the same toggle as annotations. They are hand work, they hide
together, and a fourth checkbox on that row would be the seventh widget the M6-T03 finding was about.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Store a ruler as a degenerate annotation | Refused twice by design, and the refusal is right |
| Call the table `measurements` | Two answers to "where are the measurements?" |
| Store the computed length | A second answer waiting to disagree with the endpoints |
| Compute the length in the panel | The first science in `gui/` in seven milestones |
| Show nanometres from an assumed scale | The fabrication ADR-0025 spent a milestone removing |
| A table per tool | The profile line is the same geometry; two migrations for one shape |

## Compliance

`tests/gui/test_ruler_tool.py` covers the arithmetic and its degenerate cases, the round trip
**without** a stored length, both units, an unscaled scan reading in pixels **and saying so**, a
zero-length line discarded quietly and refused loudly by the repository, undo and redo restoring the
same row, and the profile `kind` sharing the table.

## References

- ADR-0044 / ADR-0071 — the shapes a ruler is not
- ADR-0031 / ADR-0042 — the word this one had to avoid
- ADR-0072 — deriving rather than storing, the same rule one shape over
- ADR-0025 — absent is not zero, at the surface that produces the number
