# CURRENT TASK

**ID:** — (between milestones)
**Milestone:** **M4 closed 2026-08-12. M5 — GUI shell — is open.**
**Branch:** `feat/m4-application-layer` (M5 opens a new one)
**Status:** M4's record is in `docs/Progress.md`; the next task is `M5-T01`.

---

## M4, closed

**All six exit criteria met, all fifteen tasks done, ADR-0038 through ADR-0051.** The milestone
built what W1 said did not exist — *"no application layer at all: no projects, no persistence, no
jobs, no settings, no logging"* — and the golden file did not move once in fifteen tasks.

| Criterion | Where it is met |
|---|---|
| A project created, opened, populated, closed — headless | M4-T04 |
| Results round-trip through SQLite **and** the filesystem | M4-T05 |
| `DeviceManager` reports and selects on this machine | M4-T12 |
| Registry resolves `yolo` and `sam2` via `ModelDescriptor` | M4-T13 |
| Undo/redo proven on a mutating use case | M4-T08 |
| Integration tests cover the layer; no Qt anywhere | M4-T15 |

**Two defects made closable rather than closed**, and saying which is the point: **W10** (the
registry exists; `PipelineConfig` still holds a path until M5 uses it) and **W8**, which *is*
closed — the device is chosen and reaches a provider.

**Three tasks turned out not to need building**, each recorded with its reason: autosave (ADR-0046
— storage is write-through, so a service would be a timer that flushes nothing), three of five
lifecycle use cases (ADR-0041 — a function forwarding one call is a second name for the same
method), and the SQLite log sink (ADR-0051 — a log must not depend on the thing whose failure it
records).

---

## What M5 starts from

`M5-T01` — the entry point, the composition root and the `nanoscope` console script. It is the
first task that constructs the objects M4 built, in the one place PROJECT_RULES §2.7 allows, and
it inherits three obligations written down for it by name:

1. **`open_project`'s integrity report has to be shown** (ADR-0040).
2. **A confirmation dialog must count annotations before `remove_image`** (ADR-0044).
3. **A job's listener fires on a worker thread**, so the Qt adapter marshals (ADR-0043) — and the
   cancel button means *stop at the next checkpoint*, which the wording has to carry.

And the guard added in M4-T15 is waiting: nothing outside `gui/` may import Qt.
