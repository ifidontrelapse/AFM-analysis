# CURRENT TASK

**ID:** `M4-T15`
**Title:** The whole layer, headless, in one test — and a guard that keeps Qt out
**Milestone:** M4 — Application layer, last task
**Defect:** — · **ADR:** none expected; a test that only exercises existing decisions makes none
**Branch:** `feat/m4-application-layer`
**Status:** **done 2026-08-12.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is last

Fourteen tasks built the layer a piece at a time, each with its own tests. M4's sixth and final
exit criterion asks a different question:

> *Integration tests cover the whole layer; no Qt imported anywhere.*

Not "does each part work" — every part has its own file for that — but **does the layer work as one
thing**, in the order an operator uses it, with nothing but Python. That is a different test, and
it is the one that catches a seam nobody owns: a use case that needs something no adapter exposes,
a value that survives each hop and not the chain.

---

## The decisions this task has to make

**1. What does "the whole layer" mean, concretely?** One test that walks the operator's day in
order, touching every task M4 shipped:

configure logging (T14) → create a project (T04) → state a preference (T10) → register a model
(T13) → resolve a device (T12) → import a folder **as a job, with progress** (T06) → analyse (T05)
→ annotate and undo (T07, T08) → export (T11) → close → **reopen and find all of it** (T01–T03).

The reopen is the assertion that matters. Anything that only lives in memory disappears there.

**2. Is that one test or fifteen?** **One**, deliberately — and it is the only place in this
repository where that is the right shape. A long test is normally a smell, because a failure in
step nine tells you little; here the *sequence* is the subject, and splitting it into fifteen
independent tests would be fifteen more copies of what already exists in the per-task files.

**3. What does "no Qt imported anywhere" mean once M5 exists?** Not "nothing imports PySide6" —
`gui/` will, and that is its job. The honest, durable form is **nothing outside `gui/` imports
it**, checked statically over the source so it holds for code that never runs, and dynamically so
it holds for a transitive import.

Written now, while `gui/` is empty and the check is trivially true, because a guard added *after*
the first violation is a guard that has already failed once.

**4. Does anything new get built?** No. If the walkthrough needs a function that does not exist,
that is a finding to report, not a thing to quietly add — the criterion is about what M4 already
built.

---

## Scope

**In scope**

1. `tests/integration/test_whole_layer.py` — the walkthrough, and a second test proving the
   *reopened* project is what the first session left
2. `tests/unit/test_import_graph.py` — the Qt guard, static and dynamic, phrased for a world where
   `gui/` exists
3. Docs: the criterion ticked

**Out of scope**

- **New production code.** Decision 4
- **Closing the milestone.** That is the next commit, and it should be able to cite this one

---

## Definition of done

- [x] One end-to-end test covering every M4 task, ending in a reopen
- [x] The Qt guard, static and dynamic, written to survive M5
- [x] `make check` green — 828 tests, golden byte-identical
- [x] `Roadmap.md` criterion ticked; docs updated
- [x] Commit: `M4-T15: the whole layer, in the order an operator uses it`

---

## What it turned up

**B-068:** `PipelineConfig`'s default `mode` is `"segment"`, which needs a SAM2 predictor that is
not in this repository — so the most natural call in the whole project,
`PipelineConfig(detector="log")`, raises, and the **default configuration is one CI can never
execute**. Found by writing the obvious line. Filed rather than fixed: changing a default changes
what happens for every caller who omits it, including the notebooks (PROJECT_RULES §4.5).

**An in-process `sys.modules` assertion is a claim about the whole suite, not about the import it
names.** `test_ports.py` has checked since M2-T08 that importing the domain pulls in no torch, by
reading *this* process's modules — so any earlier test could break it, and today one legitimately
did: the end-to-end walkthrough probes real hardware, which imports torch on purpose. It now runs
in a subprocess, which is what M2-T09's weight check already does.

**Writing the walkthrough found no missing seam.** Fourteen tasks of separately-tested parts fitted
together on the first attempt, apart from the two findings above — which is worth recording,
because it is the thing this test existed to disprove.
