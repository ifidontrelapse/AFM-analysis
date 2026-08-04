# ADR-0013 — Standard-library `logging` instead of a `LogSink` port

- **Status:** Accepted
- **Date:** 2026-08-04
- **Affects:** `ADR-0001` §ports (amends the port list) · M2-T11 · M6 (SQLite log storage)

## Context

`ADR-0001` lists `LogSink` among the ports `core` defines and `infrastructure`
implements, and `docs/Architecture.md` §3.1 places a structured logger under
`infrastructure/logging/` writing "to file + SQLite". M2-T11 is the task that
replaces the 13 `print` calls in library code (audit D-23), so it is where that
port would have been written.

Implementing it means: a `LogSink` `Protocol` in `core/ports/`, an adapter
implementing it, a way for every science function to obtain one — a parameter, a
module global, or a container lookup — and eventually a second adapter for SQLite.

The standard library already provides all of that. `logging.getLogger(__name__)`
is the dependency-free call the domain makes; `logging.Handler` is the extension
point an adapter implements; `logging.LogRecord` is the structured payload, with
`extra=` for domain fields. Handlers are attached by the application, not by the
library — which is the same inversion `LogSink` was going to provide.

## Decision

**Library code uses `logging.getLogger(__name__)` directly. No `LogSink` port is
defined, and none is planned.**

- Every module that reported through `print` now has a module-level logger.
- No library module calls `basicConfig` or attaches a handler. Configuring
  logging is the application's decision; a library that makes it steals it.
- Messages use lazy `%`-formatting, not f-strings, so the template survives into
  the record and arguments are not rendered when the level is off.
- The SQLite log destination that `ADR-0003` and `Architecture.md` §3.1 describe
  becomes a `logging.Handler` in `infrastructure/logging/`, added in M6 when the
  database exists. It needs no new abstraction in `core`.

`ADR-0001`'s port list is amended: `LogSink` is removed from it. The other ports
are unaffected.

## Consequences

**Positive**

- One fewer abstraction, and the one deleted had exactly one candidate
  implementation — the case `ADR-0001`'s own dependency rule does not require.
- The domain gains no import beyond the standard library, so
  `tests/unit/test_import_graph.py` keeps passing without an exception.
- Every consumer of the codebase already knows how to capture the output:
  `caplog` in tests, `dictConfig` in the application, a handler in the GUI.
- Third-party libraries in the pipeline (skimage, matplotlib) log through the
  same mechanism, so one configuration covers everything.

**Negative**

- The domain now names a concrete standard-library module rather than an
  interface it owns. If logging ever has to be swapped for something that is not
  `logging`-compatible, every module changes. Judged unlikely enough to accept:
  the alternative costs an abstraction today against a swap nobody has proposed.
- `logging` is stringly-typed. Structured fields go through `extra=`, which is a
  weaker contract than a typed `LogSink.emit(event)` would have been. If the GUI
  later needs typed events, that is a different concern — an event bus, not a log.

**Neutral**

- `Architecture.md` §3.1 keeps `infrastructure/logging/`; what lives there is a
  `Handler`, not an adapter for a port we own.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Write `LogSink` as `ADR-0001` planned | One implementation, and it would wrap `logging` anyway. An interface with a single implementation that forwards to the standard library is indirection, not inversion. | A second, non-`logging` destination existed with different semantics. |
| Pass a logger into every science function | Threads a parameter through the entire numerical core for an operational concern, and every call site in the golden harness would change. | The domain were required to be import-free of the standard library, which it is not. |
| Keep `print` until the GUI needs otherwise | `print` cannot be filtered, levelled, redirected or tested, and D-23 is open precisely because it reaches users of a library. | Never. |

## Compliance

- `grep -rn "print(" nanoscope/` returns nothing; `tests/unit/test_logging.py`
  asserts it over the AST, per module, and fails on a reintroduced call.
- No library module calls `logging.basicConfig`.
- `tests/unit/test_logging.py` asserts silence when the caller configures nothing.

## References

- `ADR-0001` §ports · `docs/Architecture.md` §3.1
- Audit D-23 · `docs/TASKS.md` M2-T11
- `nanoscope/core/ports/__init__.py` — the table of which port arrives with which task
