# CURRENT TASK

**ID:** `M6-T01`
**Title:** Preprocessing an operator can see the stages of, and asked for
**Milestone:** M6 — Analysis workflow in the GUI, first task
**Defect:** — · **ADR:** **ADR-0061** (to be written)
**Branch:** `feat/m6-analysis-workflow` (from `feat/m5-gui-shell`)
**Status:** planning 2026-08-13.

---

## Why this task is first

M6's exit criterion is *load → detect → segment → measure → export, entirely through the UI*, and
**preprocessing is the step every later one stands on**: `run_analysis` calls `run_preprocessing`
before it detects anything, and the substrate it builds decides what a particle even is.

It also collects a promise ADR-0056 made by name. That ADR decided the viewer shows **the file**,
raw and unflattened, and said why in a sentence that named this task:

> *A "flatten for display" toggle is a later task with a checkbox and a name.*

This is that task, and the name is not "flatten for display" — it is **which stage of the pipeline
am I looking at**, which is a bigger and more honest question.

---

## The decisions this task has to make

**1. What can the operator change?** The three parameters `build_substrate_map` already takes, and
nothing invented.

`min_size_nm` (a physical size since ADR-0024), `manual_radius_px` (ADR-0014's radius, which is
*the* radius when given), and `opening_scale` (measured in ADR-0037, and a real trade-off: smaller
finds more particles in a dense field, larger measures radii better).

`run_preprocessing` currently accepts none of them, so they become pass-through parameters **with
the defaults it already uses** — the roadmap's rule for this whole milestone is *the UI must not
introduce its own defaults*, and the strongest form of that is a panel whose blank state produces
the byte-identical result the function produced before it existed.

**2. Is the preview live?** No — it is **asked for**, with a button, and it runs as a job.

Architecture §4.5 says anything over ~100 ms is a job, and preprocessing a 4096² scan is seconds of
NumPy. A preview that re-runs on every spinbox keystroke is a UI that fights the operator and heats
their laptop; the honest version is *change the numbers, then ask*. M5-T07 built the machinery, and
this is its second consumer — which is the point of having built it.

**3. What does the viewer show, and how does it stay honest?** The chosen stage, **named on screen**.

Raw, plane-flattened, line-flattened, substrate, or `z_result`. ADR-0056's rule was never "show the
file and nothing else" — it was *never show something the file does not contain without saying so*.
So the viewer gains a stage label, and the stage label is the whole of the compliance: an operator
comparing a height against a measurement can see which array they are looking at.

Selecting a different image drops the preview, because a substrate map computed from another scan
displayed over this one would be the worst possible version of this feature.

**4. Is a preview a result?** No, and it is not persisted.

`run_analysis` records a run (M4-T05, ADR-0042); a preview is a look at intermediate arrays. M6-T09
owns persistence, and a preview that quietly wrote rows would make "what runs does this image have?"
a question about which buttons somebody pressed.

**5. What does the panel report back?** The numbers the stage produced: the opening radius actually
used, and the Otsu size estimate with its object count.

ADR-0014 and ADR-0017 both ended on *report what was used, not what was asked for* — a manual radius
is the radius, and the estimator counts what it kept. Those are already in the result; nothing here
recomputes them.

---

## Scope

**In scope**

1. `application/use_cases/preprocessing.py` — `min_size_nm`, `manual_radius_px`, `opening_scale` as
   pass-through parameters, defaults unchanged
2. `gui/panels/preprocessing.py` — the parameters, the Preview button, and what the run reported
3. `gui/viewmodels/session.py` — `preprocess(...)` as a job, the result, and the selected stage
4. `gui/panels/viewer.py` — draw the selected stage, and **say which one it is**
5. **ADR-0061** — a preview is asked for, is not a result, and names the array it shows
6. Tests: the defaults are byte-identical to the previous call, each parameter reaches
   `build_substrate_map`, the preview runs as a job, a stage change redraws, changing image drops
   the preview, and nothing is written to the project

**Out of scope**

- **Detection** — M6-T02, and it is the next task
- **Persisting a preview** — decision 4; M6-T09
- **A live re-run on parameter change** — decision 2

---

## Definition of done

- [ ] The panel's blank state produces exactly what `run_preprocessing` produced before this task
- [ ] Every parameter reaches the science, and the panel reports what was *used*
- [ ] The viewer names the stage it is showing
- [ ] ADR-0061 + the ADR index
- [ ] `make check` green, **golden byte-identical**
- [ ] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [ ] Commit: `M6-T01: preprocessing an operator can see the stages of, and asked for`

---

## What it turned up

_(filled at the end of the task)_
