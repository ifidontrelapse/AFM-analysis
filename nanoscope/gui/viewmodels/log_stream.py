"""The application's log, on its way to a widget (M5-T08, ADR-0059).

A `logging.Handler` whose `emit` does one thing: emit a Qt signal. That is
ADR-0058 §1 for the second time, and the repetition is the point — **a log record
can be written from a worker thread**, because a job logs, so a panel appending
to a widget inside `emit()` is the crash ADR-0043 warned about in a place nobody
would look for it.

What travels is a **rendering, not the record**, which is deliberately the
opposite of what `job_changed` carries. A job's handle is sent because a progress
bar wants the *latest* state; a log line is a **fact at a moment** and must not
change between being written and being read. Formatting also has to happen on the
thread that logged: `%`-style arguments are lazy (ADR-0013), and the values they
name may be gone by the time the main thread looks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from PySide6.QtCore import QObject, Signal

#: Levels at or above this one are worth interrupting somebody for (ADR-0059 §5).
NOTIFY_LEVEL = logging.WARNING


@dataclass(frozen=True)
class LogLine:
    """One record, already rendered. Frozen because it is a fact, not a state."""

    time: str
    level: int
    level_name: str
    logger: str
    message: str

    @property
    def is_notable(self) -> bool:
        """Whether an operator should be told without having to be looking."""
        return self.level >= NOTIFY_LEVEL

    def as_text(self) -> str:
        return f"{self.time}  {self.level_name:<8} {self.logger}: {self.message}"


class LogStream(QObject):
    """The signal a panel connects to, and the handler that feeds it.

    **Two objects rather than one**, and not by preference: `class
    LogStream(QObject, logging.Handler)` was written first and works at run time,
    but `logging.Handler.emit` and **`QObject.emit`** are different methods with
    the same name — Qt's is the old-style "emit this signal by name" — so the
    subclass silently overrides a method of its own base with an incompatible
    signature. mypy found it. A `type: ignore` would have been permanent, and a
    name that means two things is worth one extra object.
    """

    #: One rendered line, delivered on the thread this object lives on.
    logged = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        #: What `app.logging.attach_view_log` is handed. Nothing else attaches
        #: it, and nothing outside `app/` may (ADR-0051).
        self.handler: logging.Handler = _SignalHandler(self)


class _SignalHandler(logging.Handler):
    """A handler whose whole body is one signal emission (ADR-0058 §1)."""

    def __init__(self, stream: LogStream) -> None:
        super().__init__()
        self._stream = stream
        self._alive = True

    def emit(self, record: logging.LogRecord) -> None:
        """Render the record here, and hand the result over.

        **Goes quiet once its window is gone.** Qt deletes the C++ object behind
        a widget before Python lets go of the wrapper, and emitting from the
        remains raises `RuntimeError` — which `logging.handleError` then prints,
        with a traceback, into the stderr of an application that is shutting
        down. Found from another test file entirely: a window that was never
        closed left this handler attached, and every later log line printed a
        logging error at whoever was watching.

        A flag rather than removing itself from the logger: `callHandlers`
        iterates the handler list, and mutating it mid-iteration silently skips
        whichever handler comes next.

        Anything else still goes to `handleError` — a handler that throws takes
        the caller's `logger.info(...)` down with it, and turns a diagnostic
        into the fault.
        """
        if not self._alive:
            return
        try:
            self._stream.logged.emit(_render(record))
        except RuntimeError:
            self._alive = False
        except Exception:  # pragma: no cover — anything else is ours to report
            self.handleError(record)


def _render(record: logging.LogRecord) -> LogLine:
    return LogLine(
        #: **Local** time, where the JSONL file records UTC (M4-T14): a file is
        #: read next to other machines' files, and a panel is read next to the
        #: clock on the operator's own wall.
        time=datetime.fromtimestamp(record.created, UTC).astimezone().strftime("%H:%M:%S"),
        level=record.levelno,
        level_name=record.levelname,
        logger=record.name,
        #: `getMessage()` applies the lazy `%` arguments, on the thread that
        #: logged them — which is the only thread where they are still true.
        message=record.getMessage(),
    )
