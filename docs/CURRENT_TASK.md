# CURRENT TASK

**ID:** `M5-T08`
**Title:** The log an operator can see, and a warning that does not need them watching
**Milestone:** M5 — GUI shell, eighth task
**Defect:** — · **ADR:** **ADR-0059**
**Branch:** `feat/m5-gui-shell`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

The Log dock is the **last placeholder in the window** — M5-T02 put a label in it saying "the log
panel arrives in M5-T08", and an empty panel is a promise only until the task it names comes round.

It is also the half of ADR-0051 nobody can reach. That ADR sent the log to two rotating JSONL files
and argued the case well; what it left is an operator who has to find
`$XDG_STATE_HOME/nanoscope/nanoscope.log` in a file manager to learn why an import refused a scan.
The lines exist and nothing shows them.

And M5-T07 built a mechanism whose second consumer is exactly this: **a log record can be written
from a worker thread** — a job logs — so a panel that appends to a widget from `emit()` is the same
crash ADR-0043 warned about, in a place nobody thinks to look for it.

---

## The decisions this task has to make

**1. How do records reach the panel?** A `logging.Handler` that emits a Qt signal, and nothing else.

The second application of ADR-0058 §1, which is the point: if the answer to "another thread touched a
widget" is a different mechanism each time, there is no answer. `emit()` does one thing.

**2. What travels on the signal — the record, or a rendering of it?** A rendering, and this is
*opposite* to M5-T07's choice.

ADR-0058 §3 sends the job **handle** because a bar wants the latest state. A log line is the
opposite kind of thing: a **fact at a moment**, which must not change between being written and
being read. Formatting also has to happen on the thread that logged, because `%`-style arguments are
lazy by ADR-0013's rule and the values may be gone by delivery. So the handler builds a frozen
`LogLine` in `emit()` and the panel only paints it.

**3. Who attaches the handler?** `app/`, as ADR-0051 requires.

*"Only `app/` attaches handlers, and they are named, so configuring twice replaces rather than
duplicates."* So `attach_view_log(handler)` joins `configure_logging` and `attach_project_log` in
`app/logging.py`, tagged `nanoscope:view`; the window asks for it and detaches on close. A handler
left attached to a dead widget is a crash on the next log line, in a process that is shutting down.

**4. What is the panel made of?** A read-only `QPlainTextEdit` with `setMaximumBlockCount`.

Qt's own bounded buffer, which is the ring buffer this needs and none of the code. Colour comes from
the tokens (ADR-0054), because a warning that looks like every other line is a warning nobody sees.

**5. What is a "notification"?** The dock counts what an operator has not looked at.

A `WARNING` or `ERROR` logged while the Log dock is hidden makes its title read **"Log (3)"**, and
showing the dock resets it. Not a toast, not an auto-raised panel that steals the focus of somebody
mid-drag: the two failure modes of desktop notifications are being missed and being resented, and a
count in a title is the version of this that is neither.

`INFO` does not notify. A notification for every ordinary line is the same as none.

---

## Scope

**In scope**

1. `gui/viewmodels/log_stream.py` — `LogLine`, and the handler that turns a record into a signal
2. `gui/panels/log_panel.py` — the bounded, coloured view
3. `app/logging.py` — `attach_view_log` / `detach_view_log`, tagged like the other two
4. `MainWindow` — the Log dock gets its panel, the title counts unseen warnings, the handler is
   detached on close
5. **ADR-0059** — a rendering not a record, who attaches, and what a notification is allowed to do
6. Tests: a record logged **from a worker thread** arriving on the main one, the bound on the
   buffer, levels coloured, the unseen counter and its reset, and the handler detaching

**Out of scope**

- **A level filter and a search box** — the buffer is bounded and the file is authoritative; a
  filter that hides lines needs an answer to "hides them from what?" that this panel does not have
  yet
- **Showing the log file's path** — a support affordance, and M9's packaging task is where "how do I
  send you my log" belongs
- **Reading the JSONL back after a restart** — the panel shows this session; the file is the history
  (ADR-0051 already says which is which)

---

## Definition of done

- [x] A record logged on a worker thread reaches the panel on the main thread, asserted by thread
- [x] The Log dock has a panel, and the placeholder list is empty
- [x] Warnings logged while the dock is hidden are counted in its title, and reset when shown
- [x] The handler is detached when the window closes
- [x] ADR-0059 + the ADR index
- [x] `make check` green — 1016 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M5-T08: the log an operator can see, and a warning that does not need them watching`

---

## What it turned up

**`logging.Handler.emit` collides with `QObject.emit`.** `class LogStream(QObject,
logging.Handler)` was written first and runs correctly — and silently overrides a method of its own
base with an incompatible signature, because Qt has an `emit` too (the old-style emit-a-signal-by-
name). mypy found it. A `type: ignore` there would never have expired, so the handler and the signal
became two objects.

**A window that is never closed leaves a handler pointing at a deleted widget** — and the next log
line prints `--- Logging error ---` with a traceback into stderr. Found from *another test file
entirely*: `test_entry_point.py`'s assertion that a refusal carries **no traceback** went red, in a
combined run only, for a handler installed by a GUI test twenty files earlier. The handler now goes
quiet on `RuntimeError` — with a flag rather than by removing itself, since `callHandlers` iterates
the list it would be mutating.

**An import wrote nothing into the project log.** Seen by reading the finished panel: ADR-0051
created that log to answer *what happened to this work*, and the first thing an operator does left
it empty. One line at the point the outcome is already known.

**`isVisible()` is `False` for every widget in a window that was never shown** — the second time this
milestone. It is the right check for the notification (a dock behind a tab is not visible either),
so the test shows the window rather than the code weakening to `isHidden()`.
