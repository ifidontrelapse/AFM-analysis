# ADR-0059 — A log line is a fact, and a notification is a count

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M5-T08)
- **Affects:** `gui/viewmodels`, `gui/panels`, `app/logging` · M5 · M6

## Context

ADR-0051 sent the log to two rotating JSONL files and argued the case well. What it left is an
operator who has to open `$XDG_STATE_HOME/nanoscope/nanoscope.log` in a file manager to find out why
an import refused a scan. The lines exist; nothing shows them.

The Log dock was also the **last placeholder** in the window. M5-T02 put a label in each dock naming
the task that would replace it, and this is that task for the last one.

And the records arrive from anywhere: **a job logs**, so a panel appending to a widget inside
`Handler.emit` is the crash ADR-0043 warned about, in the one place nobody thinks to look for a
threading bug.

## Decision

### 1. Records reach the panel as a Qt signal

The third application of ADR-0058 §1, and by now the point is the repetition: if "another thread
touched a widget" has a different answer each time, it has no answer. `emit()` builds a line and
emits a signal; Qt queues it onto the thread the panel lives on.

### 2. What travels is a rendering, not the record — the opposite of `job_changed`

ADR-0058 §3 sends the job **handle** because a progress bar wants the latest state. A log line is
the other kind of thing: a **fact at a moment**, which must not change between being written and
being read.

Formatting must happen on the logging thread anyway. `%`-style arguments are lazy by ADR-0013's
rule, and the objects they name may be gone — or, worse, changed — by the time the main thread
paints. So `LogLine` is frozen and built in `emit`.

### 3. `app/` attaches the handler, and the window lets it go

`attach_view_log` / `detach_view_log` join `configure_logging` and `attach_project_log`, tagged
`nanoscope:view` like the others, because ADR-0051's rule is that only `app/` attaches handlers and
that tagging is what makes a second window replace rather than duplicate.

The handler itself is Qt's business and stays in `gui/`; `app/logging.py` takes a
`logging.Handler` and knows nothing about widgets.

**A handler also goes quiet when its window is gone.** Qt deletes the C++ object behind a widget
before Python lets go of the wrapper, so emitting from the remains raises `RuntimeError` — which
`logging.handleError` prints, with a traceback, into the stderr of an application that is shutting
down. A flag, not self-removal: `callHandlers` iterates the handler list and mutating it
mid-iteration skips whichever handler comes next.

### 4. The panel is bounded, and the file is the history

A read-only `QPlainTextEdit` with `setMaximumBlockCount` — Qt's own ring buffer, and none of the
code. Colours come from the tokens (ADR-0054), because a warning that looks like every other line is
a warning nobody sees, and the text is HTML-escaped, since the log line most worth reading is the
one with a repr in it.

This panel is **this session**. Reading the JSONL back after a restart is not offered: ADR-0051
already decided which of the two is the record.

### 5. A notification is a count in the dock's title

A `WARNING` or `ERROR` logged while the Log dock is not visible makes its title read **"Log (3)"**;
looking at the dock resets it.

Not a toast, and not an auto-raised panel: the two ways desktop notifications fail are being
**missed** and being **resented**, and a panel that jumps in front of somebody mid-drag is the
second. A count is neither — it waits, and it is still there in ten minutes.

`INFO` does not notify. A notification for every ordinary line is the same as none.

### 6. An import is logged as well as shown

Found by looking at the finished panel: a status line lasts until the next status line, and the
first thing an operator does in this application wrote **nothing** into the project log ADR-0051
created to answer *what happened to this work*. One line, at the point the job's outcome is already
known.

## Consequences

**Positive**

- The half of ADR-0051 that needed a reader has one, and M5-T02's last placeholder is gone.
- A background failure is visible without an operator watching the status bar at the right moment.
- The panel costs nothing to keep open: bounded by construction, and painted only from the main
  thread.
- M6's analysis runs will appear here without writing any GUI code.

**Negative**

- No level filter and no search. The buffer is bounded and the file is authoritative, but an
  operator hunting one warning in two thousand lines currently scrolls. The trigger for adding one
  is somebody doing that; a filter now would need an answer to "hidden from what?" that this panel
  does not have.
- The count is per-window and not persisted. A warning logged and dismissed is gone from the badge,
  though never from the file.
- `attach_view_log` is global state by construction — one root logger, one view handler. Two windows
  are not supported, and §3's tagging makes the second one win rather than duplicate.

**Neutral**

- `LogStream` is a `QObject` **and** a separate `logging.Handler`, not one class inheriting both.
  Multiple inheritance was written first and runs correctly, but `logging.Handler.emit` and
  **`QObject.emit`** are different methods with the same name — Qt's being the old-style
  emit-a-signal-by-name — so the subclass silently overrode a method of its own base with an
  incompatible signature. mypy found it; a `type: ignore` there would have been permanent.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Append to the widget inside `Handler.emit` | A job logs; that is a widget touched from a worker thread |
| Emit the `LogRecord` itself | Lazy `%` arguments must be applied where they are still true, and a record read later is not the fact that was logged |
| Attach the handler from `gui/` | ADR-0051: only `app/` attaches handlers, and tagging is what stops two windows installing two |
| Tail the JSONL file into the panel | Reading back what this process just wrote, with a file watcher, to show what it already had |
| A toast / auto-raised dock on every error | Missed or resented; a count in a title is neither |
| Notify on `INFO` too | A notification for every line is the same as none |
| Keep every line | A memory leak with a scrollbar; the rotating file is the history |

## Compliance

- `tests/gui/test_log_panel.py::TestARecordReachesTheScreen` logs **from a worker thread** and
  asserts the delivery lands on `QApplication.instance().thread()`; it also pins the rendering
  (`%`-args applied where they were logged), HTML escaping, and the bound on the buffer.
- `TestTheNotification` pins the count, its reset, and that `INFO` does not raise it.
- `TestTheHandlerIsOwnedByTheApplication` asserts exactly one `nanoscope:view` handler for two
  windows, and none after the window closes.
- `TestAHandlerWhoseWindowIsGone` asserts a dead handler goes quiet instead of printing a logging
  error, and stops trying after the first failure.

## References

- ADR-0051 (a log must not live inside what it reports on) — the two files this reads from the other
  end, and the rule in §3
- ADR-0058 §1 and §3 — the mechanism, and the deliberate inversion in §2
- ADR-0043 — why a log record can arrive from a worker thread at all
- ADR-0013 — lazy `%` arguments, which decide where formatting happens
- ADR-0054 — the tokens the levels are coloured with
