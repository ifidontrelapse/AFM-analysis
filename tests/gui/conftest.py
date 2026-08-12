"""One `QApplication`, offscreen (M5-T02).

PROJECT_RULES §6: *"GUI tests run headless (`QT_QPA_PLATFORM=offscreen`)"*. The
variable is set **before** Qt is imported, because Qt reads it when the platform
plugin loads and setting it afterwards does nothing.

No `pytest-qt`: what these tests need is one `QApplication` that outlives the
session, and a dependency for a fixture is a dependency for nothing.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_app() -> Iterator[object]:
    """The one `QApplication` for the whole session.

    Qt allows exactly one per process and does not take kindly to it being
    destroyed and rebuilt, so it is session-scoped and never torn down.
    """
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
