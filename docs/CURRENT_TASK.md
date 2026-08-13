# CURRENT TASK

**ID:** `M7-T05`
**Title:** A distance somebody measured, in the units they can defend
**Milestone:** M7 — Annotation & metrology tools, fifth task
**Defect:** — · **ADR:** **ADR-0074**
**Branch:** `feat/m7-annotation-tools`
**Status:** **done 2026-08-14.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

Four tools in, every shape this project stores describes *a thing*. A ruler describes **a distance
between two things**, which is the first output in the whole project that no algorithm produced —
and the roadmap says so in its own risk line for M7:

> *"Manual measurements are a new output and get their own tests."*

---

## The decisions this task has to make

**1. It is not an annotation, and the reason is the same one that refused the point.**

A line has no area. ADR-0044 stores shapes with extent and refuses a zero-area one twice; a ruler
would fail both checks, and forcing it through as a degenerate box would be the invention this
project keeps declining. It gets a table.

**2. The word "measurement" is taken.**

`measurements.csv` is what an analysis run produces (ADR-0031, ADR-0042) — derived, re-runnable, and
shaped by its producer. What an operator draws by hand is none of those things, and calling both
"measurements" would make *"where are the measurements?"* a question with two answers. The table is
**`rulers`**.

**3. One table for two tools.** A profile line (M7-T06) is the same geometry read differently, so the
row carries a `kind` — and the migration happens once rather than twice.

**4. The length is `core.science`, not a widget.**

Two points and Pythagoras is arithmetic, and putting it in a panel would be the first science in
`gui/` in seven milestones. It is a new output, so it gets a module with tests: `metrology.py`.

**5. Without a scale there is no length in nanometres.**

Pixels always; nanometres only when the project recorded a scale, and **the words "scale unknown"**
otherwise. This is ADR-0025's rule arriving at the first surface that *produces* a physical number
rather than reading one.

**6. It goes through the command stack**, like every other tool since M7-T02.

---

## Scope

**In scope**

1. `core/science/metrology.py` — `distance_px`, `distance_nm`, with their degenerate cases
2. `core/entities/project.py` — `Ruler`
3. `infrastructure/storage` — schema **v8**, the `rulers` table, read and write
4. `application/commands.py` — `AddRuler`, so undo covers it
5. `gui/` — the ruler tool, the line on the canvas, and the readout
6. **ADR-0074** — a ruler is not an annotation, the word "measurement" is taken, and units are
   honest
7. Tests: the arithmetic, the round trip, an unknown scale, undo/redo, and the drawing gesture

**Out of scope**

- **The profile plot** — M7-T06 reads heights along the same geometry
- **Editing a stored ruler** — M7-T07
- **Angles and areas** — no tool has asked for them

---

## Definition of done

- [x] A drawn line stores as a ruler and reads back
- [x] Its length is right in pixels, and in nanometres only when there is a scale
- [x] Undo and redo cover it
- [x] ADR-0074 + the ADR index
- [x] `make check` green — 1231 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, `ProjectFormat.md`
- [x] Commit: `M7-T05: a distance somebody measured, in the units they can defend`

---

## What it turned up

**Undo reloaded annotations only, so undoing a ruler left the line on the canvas.** M7-T02 wired the
window's Undo label to `annotations_changed` and wrote down *why it was allowed to*: every command in
the stack mutated annotations, and the first one that did not would need its own signal. The ruler is
that command, one task later, and the note it left is what made the fix obvious rather than
mysterious.

**The gate ran against a tree that had already moved.** M7-T04's `make check` was started in the
background and M7-T05's edits landed in the same working tree while it ran, so it reported a failure
belonging to neither task. The two tasks are therefore committed together, with one clean gate over
both — stated here rather than papered over.
