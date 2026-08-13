# CURRENT TASK

**ID:** `M6-T05`
**Title:** The measurements, beside the particles they belong to
**Milestone:** M6 — Analysis workflow in the GUI, fifth task
**Defect:** — · **ADR:** **ADR-0065** · **Filed:** **B-069**
**Branch:** `feat/m6-analysis-workflow`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M6's second exit criterion is *"selecting a table row highlights the particle, and vice versa"*, and
it is the one that turns a stored measurement table into something an operator can argue with. A run
that says a particle is 14 nm tall is a number; the same number next to the particle it came from is
evidence.

`measurements_for` has been on the port since M4-T05 and has one reader — the CSV export.

---

## The decisions this task has to make

**1. What links a row to a particle?** Its **coordinates**, not its index.

The measurement table is not one row per detection: a height that is not a number is discarded
(ADR-0033), so the table is a subset and the two lists are different lengths. `x_px`/`y_px` are in
the schema's core (ADR-0031) and in every detection, so the link is a position — which is also the
only link that survives a producer that renumbers its rows.

**2. Which direction is authoritative?** Neither: the selection lives in the viewmodel.

The table asks for a particle to be selected; the canvas asks for a particle to be selected; both
listen for the answer. Two widgets talking to each other is what ADR-0057 removed, and *"vice
versa"* is exactly the case where it would come back.

**3. What does a click on the canvas mean, when dragging pans?** A press and a release in the same
place.

The view drags to pan (M5-T05), so a click is a *release without movement*. Three pixels of
tolerance, because a mouse moves under a finger.

**4. What does the table show?** The stored table, as it is.

Column names and dtypes are the producer's (ADR-0031) — a panel that renames them for display is a
second vocabulary, and a reader of the exported CSV would then be reading different words for the
same measurement.

---

## Scope

**In scope**

1. `gui/viewmodels/session.py` — the measurement table for the current run, the selected particle,
   and the signals for both
2. `gui/panels/measurements.py` — the table, and what it does with a selection
3. `gui/panels/viewer.py` — the highlighted item, and the click that picks one
4. **ADR-0065** — coordinates as the link, selection in the viewmodel, a click that is not a drag
5. Tests: the table matches the stored one, a row selects the particle, a click selects the row, a
   run with no table says so, and the selection clears with the image

**Out of scope**

- **Sorting and filtering the table** — the stored order is the producer's; sorting is a view state
  nothing has asked for yet
- **Editing a measurement** — a measurement is derived; the thing an operator edits is an
  annotation (M7)
- **Statistics and histograms** — M6-T06

---

## Definition of done

- [x] The table shows the run's stored measurements, with the producer's own column names
- [x] Selecting a row highlights the particle; clicking the particle selects the row
- [x] A detect-only run says it measured nothing rather than showing an empty grid
- [x] ADR-0065 + the ADR index
- [x] `make check` green — 1108 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M6-T05: the measurements, beside the particles they belong to`

---

## What it turned up

**`particle_id` means two different things depending on which producer wrote the row.** The baseline
producer writes the *blob's* index; the segmentation producers write the index of the row being
appended, which renumbers after every discard. One column, two meanings, in the schema ADR-0031 built
to stop exactly that — filed as **B-069**, and the reason this task links rows to particles by
**coordinates** instead.

**Tabbing a dock changed what an M5-T08 test meant.** Putting Measurements in front of the Log dock
made `log_dock.show()` insufficient — a dock behind a tab is *not visible*, which is precisely the
semantics the unseen-warning count wants, and a trap for a test that only calls `show()`.

**An absolute size assertion became order-dependent.** The measurements table's minimum width pushed
the window's minimum past the 640 the layout test asked for, and the failure only appeared in a full
run. The test now compares the second window against **what the first actually got**, which is what
it always meant.
