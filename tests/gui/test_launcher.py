"""The event loop, started and stopped (M5-T02, ADR-0053).

`tests/gui/test_main_window.py` builds a window and pokes it. This file does the
one thing that file cannot: it runs `gui.launcher.run` for real — window shown,
`QApplication.exec()` entered, exit code returned — with a timer that quits the
loop a moment later.

It exists because M5-T02's `--gui` branch stopped being a stub, and the test
that used to check it **hung** rather than failed. A hang is the worse of the
two, and the fix is a test that enters the loop on purpose and knows how to
leave.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.core.values import Modality
from nanoscope.gui.launcher import run
from nanoscope.gui.main_window import MainWindow
from nanoscope.infrastructure.storage import SqliteProjectRepository

pytestmark = pytest.mark.usefixtures("qt_app")


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Nanoscope]:
    with Nanoscope(settings_path=tmp_path / "settings.json") as container:
        yield container


def quit_soon(milliseconds: int = 50) -> None:
    """Leave the event loop shortly after it starts.

    A timer rather than a thread: the quit has to happen *on* the Qt thread, and
    50 ms is long enough for the loop to be running and short enough that a
    suite of these costs nothing.
    """
    QTimer.singleShot(milliseconds, QApplication.instance().quit)


def test_the_loop_runs_and_exits_cleanly(app: Nanoscope) -> None:
    quit_soon()

    assert run(app) == 0


def test_it_opens_the_project_it_was_given(app: Nanoscope, tmp_path: Path) -> None:
    """`nanoscope --gui --project X` opens X once, in the window, rather than
    printing it headlessly first."""
    root = tmp_path / "Gold on mica"
    with SqliteProjectRepository.create(root, "Gold on mica") as repo:
        (repo.root / "images" / "scan.spm").write_bytes(b"AFM")
        repo.add_image("images/scan.spm", modality=Modality.AFM)

    quit_soon()
    assert run(app, root) == 0

    assert app.repository is not None
    assert app.repository.name == "Gold on mica"


def test_a_refused_project_does_not_stop_the_window_opening(
    app: Nanoscope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator whose last project moved should get a window with a message,
    not a process that exits before they can choose another one."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)
    quit_soon()

    assert run(app, tmp_path / "not-a-project") == 0
    assert app.repository is None


def _launched_window(before: list[MainWindow]) -> MainWindow:
    """The window `run` just made, told apart from any left over by an earlier test.

    Not `QApplication.activeWindow()`: offscreen keeps previous tests' windows
    alive and one of them answers that question, which had this reading a
    different window's state.

    `before` holds the widgets themselves rather than their `id()`s, which was
    the first attempt and is a trap: CPython reuses an address once the object
    at it is collected, so a genuinely new window could be "seen before". Keeping
    the references also keeps them alive, which is what makes the comparison mean
    anything.
    """
    fresh = [
        widget
        for widget in QApplication.topLevelWidgets()
        if isinstance(widget, MainWindow) and all(widget is not old for old in before)
    ]
    assert len(fresh) == 1, fresh
    return fresh[0]


def _maximised_during_run(app: Nanoscope) -> bool:
    """Run the loop, and report whether the window it showed came up maximised."""
    before = [w for w in QApplication.topLevelWidgets() if isinstance(w, MainWindow)]
    seen: list[bool] = []
    QTimer.singleShot(
        50,
        lambda: seen.append(
            bool(_launched_window(before).windowState() & Qt.WindowState.WindowMaximized)
        ),
    )
    quit_soon(100)

    assert run(app) == 0
    assert len(seen) == 1
    return seen[0]


def test_a_first_run_is_maximised(app: Nanoscope) -> None:
    """The window an operator sees the first time they start the application.

    With nothing stored, Qt sizes it from its `sizeHint` — a viewer surrounded
    by nine docks — and the hint is a window too small to work in. The decision
    lives here rather than in the constructor because it is a decision about
    *showing* the window.
    """
    assert _maximised_during_run(app)


# There is no launcher test for the *other* branch — a stored size that fits, so
# `run` calls `show()` rather than `showMaximized()`. It cannot be staged here:
# the offscreen platform's screen is 800x800 and this window's minimum is taller
# than that once the theme is applied, so on this platform the application is
# always right to maximise. The decision itself is
# `MainWindow._reject_a_layout_that_does_not_fit`, and
# `tests/gui/test_main_window.py` states both of its outcomes with the sizes
# written out; what is left here is the two-line `if` that reads the answer.
