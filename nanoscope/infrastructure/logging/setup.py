"""JSON Lines records, in rotating files (M4-T14, ADR-0051).

Two destinations, because there are two questions. The application log answers
*what did this application do* — including everything before a project is open,
and every failure to open one. The project log answers *what happened to this
work*, and travels with the directory, which is what ADR-0003 set `logs/` aside
for.

**There is no SQLite sink**, and that is a decision rather than an omission
(ADR-0051): a log must not depend on the thing whose failure it records, and the
most valuable lines this application will ever write are the ones about a
database it could not open.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

#: 5 MB, three of them. Not a number worth agonising over, but worth stating: an
#: unbounded log on a laptop is a disk-full bug that arrives months later.
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

LOG_FILE_NAME = "nanoscope.log"
_PROJECT_LOG_DIRECTORY = "logs"

#: Attributes `logging` puts on every record. Anything else a caller passed as
#: `extra=` is theirs, and goes into the object beside the message — which is
#: what makes the log *structured* rather than a formatted string.
_STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per line: `grep` still works, and `jq` works properly.

    A format string would make a GUI panel parse a sentence back into fields it
    already had. This keeps them.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            #: `getMessage()` applies the lazy `%` arguments ADR-0013 requires,
            #: so the template and its values arrive together and neither is
            #: rendered when the level is off.
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        payload.update(
            {key: _plain(value) for key, value in record.__dict__.items() if key not in _STANDARD}
        )
        return json.dumps(payload, ensure_ascii=False)


def _plain(value: object) -> object:
    """Whatever `extra=` carried, reduced to something JSON can hold.

    A log line must never fail because somebody logged an object: the record is
    the thing being kept, and a repr is more useful than an exception.
    """
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    return repr(value)


def application_log_path() -> Path:
    """`$XDG_STATE_HOME/nanoscope/nanoscope.log`, or the conventional default.

    State rather than config (M4-T10's settings): a log is generated data that a
    user does not edit, and XDG has a directory for exactly that.
    """
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "nanoscope" / LOG_FILE_NAME


def make_rotating_handler(path: Path | str, level: int = logging.INFO) -> RotatingFileHandler:
    """A rotating JSONL handler at `path`, creating its directory if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        destination, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(JsonLinesFormatter())
    handler.setLevel(level)
    return handler


def make_project_handler(
    project_root: Path | str, level: int = logging.INFO
) -> RotatingFileHandler:
    """The handler that writes into a project's own `logs/`.

    What an operator attaches to a bug report about *that project*, and what
    stays with the work when the directory is copied.
    """
    return make_rotating_handler(Path(project_root) / _PROJECT_LOG_DIRECTORY / LOG_FILE_NAME, level)
