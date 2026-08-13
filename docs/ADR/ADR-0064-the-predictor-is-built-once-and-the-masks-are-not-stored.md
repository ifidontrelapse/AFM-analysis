# ADR-0064 — The predictor is built once, and the masks are not stored

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M6-T04)
- **Affects:** `app/container`, `core/entities/project`, `gui` · M6

## Context

M6-T02 disabled every `segment` row with a sentence naming this task. It is also the last unwired
half of ADR-0050: the registry hands back **factories**, and `resolve()` had no caller outside its
own tests.

Two things were already decided and had to be respected rather than re-opened: masks are **not
persisted** (ADR-0042 — SAM2's weights are outside the gate, so the format would have been written
blind), and `gui/` may not write the model names (PROJECT_RULES §2.5).

## Decision

### 1. The composition root builds the predictor, lazily, once

`Nanoscope.segmentation_predictor()`: a `SEGMENT`-task model in the open project → `resolve()` →
the factory, with the device `select_device` would choose (ADR-0049). Cached, because loading
weights is seconds of disk and GPU; dropped when the project closes, because a predictor belongs to
the project whose model produced it.

### 2. "Can this project segment?" is answered by a registered model, not a built one

The matrix's `has_predictor` comes from `list_models()`. ADR-0050 made the registry cheap on purpose;
filling a combo box must not read weights. The predictor is constructed **inside the job**, on the
worker thread, where the rest of the analysis already runs.

### 3. There is no new panel

The detection panel offers modes from the matrix and `segment` is one of them. A second panel would
carry its own detector and mode choices — the duplication ADR-0062 was written to prevent, one task
later. What this task adds is the mode becoming *available* and the masks becoming visible. **The
task list asked for a panel; the panel already existed.**

### 4. The mask parameters are not offered, and the reason is structural

`PipelineConfig`'s fields for them are named after the framework, **the golden records that class's
field names**, and a widget setting them would have to write the name §2.5 forbids. Renaming is a
golden-moving change and gets its own commit (ADR-0010). They stay at the defaults every call before
this one used.

### 5. The masks ride on the run, in memory only

`AnalysisRun.masks` is empty on everything the repository returns and filled on the run just
computed. That is the truth about what is stored: an overlay drawn from a re-read run would be
showing something the project cannot restore.

### 6. A mask is drawn as an outline

Filled would hide the pixels it describes, and those pixels are the measurement. The outline is
built from the mask's own row spans rather than a contour finder — `skimage` is `infrastructure` and
`gui/` may not import it, and a per-row span is exact where a polygon is a second shape.

The toggle **disappears when there are no masks**: a control for something that does not exist
teaches an operator to ignore the row it sits in.

## Consequences

**Positive** — the matrix's last disabled row can be enabled by registering a model; ADR-0050's
factories have their first real caller; the mask overlay says what is true about persistence by
having nothing to draw for an old run.

**Negative** — masks vanish when the image is deselected, and there is no way to get them back except
by running segmentation again. That is ADR-0042's deferral still standing, and the trigger for
reversing it is a format decision plus a migration. The mask parameters remain unreachable until the
rename in §4 is done as its own commit. And the outline is built row by row in Python: fine for a
few hundred masks on a phantom, and the thing to replace first if a real scan stutters.

**Neutral** — segmentation is exercised only by a **stub predictor** registered through the registry.
There are no weights here or in CI, so the alternative is not testing the path at all (M3-T14's
precedent, and the same reason its golden delta was zero by construction).

## Alternatives considered

| Alternative | Why not |
|---|---|
| Build the predictor when the panel loads | Reads weights to fill a combo box |
| A separate segmentation panel | A second copy of the detector and mode choice — ADR-0062's own case |
| Persist masks now | A format decision with a migration, made blind without weights (ADR-0042) |
| Rename the framework-named config fields | Moves golden-recorded field names; its own commit (ADR-0010) |
| Fill the mask overlay | Hides the pixels that are the measurement |
| Trace contours with `skimage` | `gui/` may not import infrastructure, and a polygon is a second shape |

## Compliance

`tests/gui/test_segmentation.py` registers a stub factory through the registry and asserts: the mode
is refused without a model and selectable with one; asking about availability loads nothing; the
predictor is built **once** per project and dropped on close; a re-read run has **no** masks while the
computed one does; and the overlay draws one outline per mask, hides on demand, and takes its toggle
away when there is nothing to show.

## References

- ADR-0050 (a model is a record, and the registry hands back factories) — §1, §2
- ADR-0042 — why masks are not stored, quoted rather than revisited
- ADR-0062 — the panel that already exists, and the duplication §3 avoids
- ADR-0049 — the device the factory is handed
