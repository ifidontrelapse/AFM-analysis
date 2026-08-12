"""Where `QApplication` is created, and the only place Qt starts (M5-T02).

`app/main.py` must not import PySide6 — M4-T15's guard says so for every module
outside `gui/`, and the guard is right: the headless entry point is the one CI
runs and the one that works when a project will not open. So the `--gui` branch
imports **this**, inside the function, and Qt is loaded when a window is asked
for and never otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


def run(app: Nanoscope, project: Path | str | None = None, argv: Sequence[str] = ()) -> int:
    """Show the window and run the event loop. Returns the process exit code.

    Args:
        app: the composition root, already built. The window is handed one; it
            never constructs one (Architecture §2.3).
        project: a project to open once the window is up, from `--project`.
        argv: passed to `QApplication` so Qt's own flags (`-style`, `-platform`)
            keep working.

    Returns:
        Qt's exit code, which is what the process should return.
    """
    qt = QApplication.instance() or QApplication(list(argv))
    window = MainWindow(app)

    if project is not None:
        window.open_project(project)

    window.show()
    logger.info("window shown")
    return int(qt.exec())
