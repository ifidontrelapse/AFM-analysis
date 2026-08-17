# ADR-0079 — The measurement vocabulary is a contract, and a test holds it

- **Status:** Accepted
- **Date:** 2026-08-17
- **Deciders:** operator + agent (M7-T10)
- **Affects:** `docs/`, `core/science/measurement` · M7 · M8 · M9

## Context

Four producers write measurement tables, two hand tools produce numbers no algorithm produced, and
what each column *means* has lived in the docstring of the function that computes it. ADR-0031 already
made the columns one declaration; their **semantics** were never written down in one place.

M8 will compare a trained model against these numbers, M9 will put them in a manual, and an operator
will put them in a paper. A column called `height_nm` is not a measurement until somebody writes down
what it is the height of.

## Decision

### 1. One reference document, `docs/Measurements.md`

The questions an operator actually asks are about relationships between producers — *is this height
comparable to that one?* — and a docstring can only describe the function it sits on. The document
covers units and coordinate conventions, every column, what is dropped and why, the two hand tools,
and the places two producers answer the same question differently.

### 2. A test holds it to the schema

Every column `measurement_columns()` can declare must appear in the document, and the document may
name no column the schema lacks. M5-T03's rule applied to prose: *the rule and its enforcement ship
together, or only the rule does.* A column added in M8 fails the gate until somebody writes down what
it means — which is the only mechanism that has ever kept a document current in this project.

The check is deliberately about **vocabulary, not sentences**: a test that asserted paragraphs would
be rewritten to match whatever the document said, which is enforcement in name only.

### 3. `height_nm` is one column and two estimators, and `method` is the discriminator

Both producers compute *peak − baseline*. Neither the peak nor the fallback is the same quantity: the
baseline producer takes its peak over a **circular mask built from the detector's sigma** and falls
back to the **global substrate median**; the SAM2 producer takes it over the **eroded real mask** and
**skips the particle** instead. Neither is wrong. A table that mixes them without saying so is, and
`method` is the column that says so.

The same holds for `area_px`, and there it is sharper: on one path it is a **disk drawn from the
detector's estimate** and on the other a **measurement of the particle**.

### 4. The project reports radii, not diameters

The task that produced this document is named *"height, diameter, distance, aspect ratio"* and there
is no diameter column: there is `radius_px` / `radius_nm` — the **equivalent-area** radius of a
measured mask — and `detector_radius_nm`, which is what the detector thought before anything was
measured. The document states the conversion rather than the schema growing a column that is `2 ×`
another one, which would be a second answer able to disagree with the first (ADR-0074's rule, applied
to a table).

### 5. Two degenerate cases are documented and filed, not fixed here

Reading the code for this document found `aspect_ratio` reporting **1.0 — the value meaning *a
circle*** — for a mask with no minor axis, and `circularity` reporting **12.57** for a single pixel,
both through a `... if x > 0 else 1.0` guard. Measured, both. They change what a stored number means,
so they need operator sign-off and their own commit (PROJECT_RULES §4.4, §4.5) — **B-071**, with
**B-072** for the deprecated skimage properties on the same expression.

Documenting a defect is not endorsing it: the document says what the number is *today*, names the
defect id, and tells a reader how to read the value until it is fixed.

## Consequences

**Positive** — the numbers can be defended outside this repository; M8's evaluation and M9's manual
have one source; the traps that produce wrong conclusions from right numbers (`particle_id`,
`area_px`, `baseline_source == "global"`) are written where a reader meets the column.

**Negative** — a second place to update when a producer changes. The test makes the *columns*
impossible to forget and the prose still possible to leave stale; that is the honest limit of a check
that does not read English.

**Neutral** — the document describes what exists, including two filed defects, so it will need an
edit when they are fixed. That edit is part of their commits.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Docstrings only | Cannot say how two producers differ; nobody reads four modules to compare two columns |
| A section in `ProjectFormat.md` | That document is a *format* contract — where files are, not what numbers mean |
| Wait for M9's user manual | M8 consumes these numbers first, and a manual is a guide rather than a reference |
| Fix B-071 in this commit | A numerical-output change bundled into a documentation task, against §4.4 twice over |
| Add a `diameter_nm` column | A second answer that can disagree with `radius_nm` (ADR-0074's argument) |
| No test | The document is the one artefact nothing executes, so it drifts first and silently |

## Compliance

`tests/unit/test_measurement_docs.py` parametrises over every column of every block and asserts each
is named in `docs/Measurements.md`; asserts the document invents no column; and carries its own
can-this-fail check. Verified by adding a `volume_nm3` column to the schema — the suite went red on
exactly that column name — and removing it again.

## References

- ADR-0031 — one measurement schema, and the four it replaced
- ADR-0033 / ADR-0025 — the two earlier removals of "a constant standing in for an undefined value"
- ADR-0074 / ADR-0075 — the hand tools this document's §7 describes
- B-069, B-071, B-072 — the defects the document names where a reader meets the column
