"""Where log records go, once somebody decides (M4-T14, ADR-0051).

M2-T11 gave every module a logger and attached nothing, on purpose: configuring
logging is the application's decision, and a library that makes it steals it
(ADR-0013). This package is the destination half — formatter and handlers — and
`app/logging.py` is the composition root that installs them.
"""

from nanoscope.infrastructure.logging.setup import (
    JsonLinesFormatter,
    application_log_path,
    make_project_handler,
    make_rotating_handler,
)

__all__ = [
    "JsonLinesFormatter",
    "application_log_path",
    "make_project_handler",
    "make_rotating_handler",
]
