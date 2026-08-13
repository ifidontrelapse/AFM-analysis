# ADR-0067 — An export names its scope, and lands in the project

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M6-T07)
- **Affects:** `gui/main_window`, `gui/viewmodels` · M6

## Context

M6's first exit criterion is *load → detect → segment → measure → **export CSV**, entirely through
the UI*. `export_measurements` has been in `application` since M4-T11 (ADR-0048) with tests as its
only callers: the operator's way to get a table out of this project was a Python prompt.

## Decision

### 1. Two scopes, named in the menu

*Export This Run* and *Export All Measurements*. ADR-0048 built the second deliberately — *"statistics
across a dataset is why the measurements exist"* — and a single item that silently meant one of them
is one somebody uses wrong exactly once, on the export they then analyse.

### 2. The file goes into the project's `exports/`, with the name the use case chooses

Not a file dialog. An export is **part of the project** (ADR-0003's layout) and is timestamped so
today's does not replace yesterday's (ADR-0048). Asking an operator where to put a file they have not
seen yet is asking them to invent a filing system per export; what the window owes them instead is
**where it went**, which the status bar says.

### 3. Nothing to export is the use case's own sentence

`export_measurements` raises rather than writing headers with no rows, because such a file says *"we
measured and found nothing"* — a different statement. The window shows that sentence. It does **not**
pre-empt it by disabling the action for a detect-only run: a disabled control says less than the
refusal does, and the operator learns nothing about why.

### 4. It runs as a job

Reading every stored table in a project is disk. The runner has been there since M5-T07, and this is
its fourth consumer.

## Consequences

**Positive** — the criterion's last step is reachable; both scopes are visible, so the dataset export
is not a hidden feature; the file lands where a backup tool and a file manager both already look.

**Negative** — an export cannot be written outside the project. That is deliberate (an export that
leaves the project is a copy, and copying is what a file manager is for), and the trigger for
revisiting it is an operator who wants one written straight to a share.

**Neutral** — the export's shape is ADR-0048's, provenance first. Choosing columns is not offered,
because the provenance is what makes a column of heights mean something on somebody else's desktop.

## Alternatives considered

| Alternative | Why not |
|---|---|
| One "Export…" item | Silently means one scope; used wrong once, on the export that gets analysed |
| A file dialog | Asks the operator to invent a filing system per export; ADR-0048 already names the file |
| Disable the action for a detect-only run | Says less than the refusal, and teaches nothing |
| Export on the main thread | Reads every stored table in the project |

## Compliance

`tests/gui/test_export_ui.py` asserts each scope exports what it names (one image versus both), that
the file lands under `exports/` and the status bar carries its relative path, that a detect-only run
is refused **in the use case's own words**, and that the window enables each action when it applies.

## References

- ADR-0048 (an export is not a copy of the stored table) — the scopes, the name, and the refusal
- ADR-0003 — why `exports/` is inside the project
- ADR-0043 / ADR-0058 — the job and its marshalling
