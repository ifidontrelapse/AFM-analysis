"""The composition root's half of logging (M4-T14, ADR-0051).

The only place in this project that attaches a handler. ADR-0013 fixed that
rule — *"configuring logging is the application's decision; a library that makes
it steals it"* — and PROJECT_RULES §2.7 says which layer the application is.

Two calls, in the order they happen: `configure_logging()` at startup, before
anything can fail, and `attach_project_log(root)` when a project opens, so what
happens to that work is written down beside it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nanoscope.infrastructure.logging import (
    application_log_path,
    make_project_handler,
    make_rotating_handler,
)

#: Marks the handlers this module installed, so configuring twice replaces
#: rather than duplicates — which is what a test suite, a restarted GUI and an
#: operator reopening a project all do.
_TAG = "nanoscope"


def configure_logging(
    level: int = logging.INFO,
    path: Path | str | None = None,
    *,
    console: bool = False,
) -> Path:
    """Send the application's log to a rotating JSONL file. Returns where.

    Called once, at startup, **before** anything can fail — the lines about a
    database that will not open are the ones worth having, and they happen
    before a project exists (ADR-0051).

    Args:
        level: the root level. `INFO` by default; a `--debug` flag is M5's.
        path: where to write, defaulting to `$XDG_STATE_HOME/nanoscope/`.
        console: also echo to stderr, for a headless run or a terminal session.

    Returns:
        The file being written to, so a caller can tell an operator where it is.
    """
    destination = Path(path) if path is not None else application_log_path()

    root = logging.getLogger()
    root.setLevel(level)
    _remove_tagged(root, f"{_TAG}:application")

    handler = make_rotating_handler(destination, level)
    handler.set_name(f"{_TAG}:application")
    root.addHandler(handler)

    if console:
        stream = logging.StreamHandler()
        stream.setLevel(level)
        stream.set_name(f"{_TAG}:console")
        _remove_tagged(root, f"{_TAG}:console")
        root.addHandler(stream)

    return destination


def attach_project_log(project_root: Path | str, level: int = logging.INFO) -> Path:
    """Also write into this project's `logs/`, replacing any previous project's.

    One project's log must not continue in another's file, so opening a second
    project detaches the first — which is why the handler is tagged rather than
    merely added.

    Returns:
        The file being written to.
    """
    root = logging.getLogger()
    _remove_tagged(root, f"{_TAG}:project")

    handler = make_project_handler(project_root, level)
    handler.set_name(f"{_TAG}:project")
    root.addHandler(handler)
    return Path(handler.baseFilename)


def detach_project_log() -> None:
    """Stop writing into a project's log. What closing one does."""
    _remove_tagged(logging.getLogger(), f"{_TAG}:project")


def _remove_tagged(logger: logging.Logger, name: str) -> None:
    for handler in [h for h in logger.handlers if h.name == name]:
        logger.removeHandler(handler)
        handler.close()
