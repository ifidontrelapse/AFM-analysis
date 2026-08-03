# ADR-0002 — Qt6 / PySide6 desktop application, dark theme only

- **Status:** Accepted
- **Date:** 2026-08-03
- **Affects:** `gui/`, `resources/` · M5–M8

## Context

The product is a Linux desktop application for scientific image analysis, used by an
operator on their own machine with their own data and their own GPU. The workloads are
local: hundreds of megabytes of scans, model weights of 137–324 MB, GPU inference, and
long-running training.

The previous direction was a browser client (`frontend/`) talking to an HTTP backend that
was never written. That architecture is a poor fit for this workload: uploading scans to
a server the user is also running, re-encoding images for transport, and losing direct
GPU and filesystem access — all to gain a delivery channel nobody asked for.

The reference applications the operator named — Label Studio, VSCode, napari, Gwyddion —
are all dense, dockable, keyboard-driven analysis environments.

## Decision

Build the UI with **Qt6 via PySide6**.

- **Dark theme only.** One theme, defined by design tokens plus a QSS stylesheet in
  `gui/theme/`. No runtime theme switching, no light variant.
- **Minimalistic, dockable layout** in the tradition of VSCode and napari: a main canvas,
  dockable side panels, a command-driven interface, keyboard-first where possible.
- **The GUI contains no business logic.** Views render and emit; viewmodels hold view
  state and call use cases. A widget never decides what to compute, which device to use,
  or how to measure.
- Long operations run as cancellable jobs off the UI thread (M5-T07). The UI never blocks.

## Consequences

**Positive**

- Direct filesystem and GPU access; no upload step, no serialization tax on 512² float
  arrays.
- Mature widget set for exactly this application shape: docks, tool palettes, graphics
  views, tables, undo framework.
- PySide6 is Python, so the UI shares the language and the type checker with the domain.
- Offline by construction — a hard requirement for lab machines.
- LGPL licensing (PySide6) is compatible with the project's Apache-2.0 license.

**Negative**

- Qt is a large dependency with its own idioms; contributors need to learn signals/slots,
  the model/view framework, and thread affinity rules.
- GUI testing is harder than testing a pure function; it needs `pytest-qt` and an
  offscreen platform in CI.
- Packaging a Qt application for Linux distribution is a real task (M9-T03), not a
  `pip install`.
- Dark-theme-only will be an accessibility complaint eventually. Accepted for v1;
  recorded as B-048.

**Neutral**

- The existing React client becomes obsolete (ADR-0007).

## Alternatives considered

| Alternative | Why not |
|---|---|
| Web client + local FastAPI backend | Two delivery layers, an HTTP contract to keep in sync, image re-encoding, and a serialization boundary for masks and DataFrames — all for a single-user local application. The half-built version of this is exactly what we are parking. |
| Electron / Tauri | Second language and runtime; the numerical core is Python, so every operation would still cross a process boundary. |
| napari plugin | Fastest path to a viewer, but napari owns the application shell — no control over projects, training, model management, or packaging. Good inspiration, wrong container. |
| Tkinter / Dear PyGui | Insufficient for dockable, tool-heavy scientific UIs with tables and complex canvases. |
| Qt via PyQt6 | GPL/commercial licensing. PySide6 is LGPL and Qt's official binding. |

## Compliance

- Nothing outside `gui/` and `app/` imports PySide6.
- `gui/` never imports `core.science` or `infrastructure` — enforced by lint (M5-T11).
- Colours and spacing come from the theme tokens; no hardcoded hex in widget code.
- GUI smoke tests run headless (`QT_QPA_PLATFORM=offscreen`) in CI.

## References

- `systempromt.md` (GUI section)
- `docs/Architecture.md` §3
- ADR-0007
