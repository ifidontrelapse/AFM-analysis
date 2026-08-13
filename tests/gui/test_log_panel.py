"""The log, on screen (M5-T08, ADR-0059).

ADR-0051 sent the log to two rotating JSONL files and left an operator who had to
find `$XDG_STATE_HOME` in a file manager to read it. This is the other end.

The test that matters is the first one: **a record logged on a worker thread**
reaches the panel on the main thread. A job logs, so this is not hypothetical —
it is ADR-0043's crash in a place nobody would look for it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from nanoscope.app.container import Nanoscope
from nanoscope.app.logging import attach_view_log, detach_view_log
from nanoscope.gui.main_window import LOG_DOCK, MainWindow
from nanoscope.gui.panels.log_panel import MAX_LINES, LogPanel
from nanoscope.gui.viewmodels.log_stream import LogLine, LogStream

pytestmark = pytest.mark.usefixtures("qt_app")

logger = logging.getLogger("nanoscope.test.logpanel")


@pytest.fixture(autouse=True)
def loud_root() -> Iterator[None]:
    """Everything is recorded, and nothing is left attached afterwards.

    A view handler surviving a test would keep painting into a widget the next
    test has already deleted — the same failure `detach_view_log` exists for.
    """
    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.DEBUG)
    yield
    root.setLevel(before)
    detach_view_log()


@pytest.fixture
def stream() -> LogStream:
    model = LogStream()
    attach_view_log(model.handler)
    return model


def settle() -> None:
    """Let Qt deliver what a worker thread queued."""
    QApplication.processEvents()


class TestARecordReachesTheScreen:
    def test_a_line_logged_here_appears(self, stream: LogStream) -> None:
        panel = LogPanel(stream)

        logger.info("imported %d file(s)", 12)
        settle()

        assert "imported 12 file(s)" in panel.text
        assert "INFO" in panel.text

    def test_a_line_logged_on_a_worker_thread_arrives_on_the_main_one(
        self, stream: LogStream
    ) -> None:
        """The reason this is a signal and not a direct call. Without the queued
        connection the panel is painted from a worker thread, which is a crash
        that happens later, somewhere else (ADR-0043, ADR-0058 §1)."""
        panel = LogPanel(stream)
        threads: list[QThread] = []
        stream.logged.connect(lambda _line: threads.append(QThread.currentThread()))

        worker = threading.Thread(target=logger.warning, args=("from a worker",))
        worker.start()
        worker.join(5.0)
        settle()

        assert threads == [QApplication.instance().thread()]
        assert "from a worker" in panel.text

    def test_the_message_is_rendered_where_it_was_logged(self, stream: LogStream) -> None:
        """`%`-style arguments are lazy (ADR-0013), so the values have to be
        applied on the thread that had them — not when the panel gets round to
        painting."""
        lines: list[LogLine] = []
        stream.logged.connect(lines.append)

        logger.info("scale %s nm/px", 1.95)
        settle()

        assert lines[-1].message == "scale 1.95 nm/px"
        assert lines[-1].logger == logger.name

    def test_html_in_a_message_is_shown_and_not_interpreted(self, stream: LogStream) -> None:
        """The log line most worth reading is the one with a repr in it."""
        panel = LogPanel(stream)

        logger.error("cannot read <ndarray of shape (4,)>")
        settle()

        assert "<ndarray of shape (4,)>" in panel.text

    def test_the_buffer_is_bounded(self, stream: LogStream) -> None:
        """A panel that keeps every line of a day's work is a memory leak with a
        scrollbar; the file has the rest (ADR-0051)."""
        panel = LogPanel(stream)

        for index in range(MAX_LINES + 50):
            panel.append(LogLine("00:00:00", logging.INFO, "INFO", "t", f"line {index}"))

        assert "line 0" not in panel.text
        assert f"line {MAX_LINES + 49}" in panel.text
        assert panel.view.blockCount() <= MAX_LINES


class TestTheNotification:
    """A count in a dock title: not missed, and not resented (ADR-0059 §5)."""

    @pytest.fixture
    def window(self, tmp_path: Path) -> Iterator[MainWindow]:
        """Shown, offscreen. `isVisible()` is `False` for every widget in a
        window that was never shown — which is the right semantics for the
        notification (a dock behind a tab is not visible either) and a trap for
        a test that skips `show()`."""
        with Nanoscope(settings_path=tmp_path / "settings.json") as container:
            window = MainWindow(container)
            window.show()
            settle()
            yield window
            window.close()

    def test_a_warning_while_the_dock_is_hidden_is_counted(self, window: MainWindow) -> None:
        window.log_dock.hide()

        logger.warning("the substrate mask is empty")
        logger.error("no reader for scan.tif")
        settle()

        assert window.log_dock.windowTitle() == f"{LOG_DOCK} (2)"

    def test_an_ordinary_line_does_not_notify(self, window: MainWindow) -> None:
        """A notification for every line is the same as none."""
        window.log_dock.hide()

        logger.info("opened project 'Gold on mica'")
        settle()

        assert window.log_dock.windowTitle() == LOG_DOCK

    def test_looking_at_it_clears_the_count(self, window: MainWindow) -> None:
        window.log_dock.hide()
        logger.warning("something")
        settle()

        window.log_dock.show()
        #: `raise_` as well as `show`: M6-T05 tabbed this dock with the
        #: measurements, and a dock behind a tab is **not visible** — which is
        #: the semantics the count wants and a trap for a test that only shows it.
        window.log_dock.raise_()
        settle()

        assert window.log_dock.windowTitle() == LOG_DOCK

    def test_a_warning_while_it_is_open_is_not_counted(self, window: MainWindow) -> None:
        window.log_dock.show()
        window.log_dock.raise_()
        settle()

        logger.warning("something")
        settle()

        assert window.log_dock.windowTitle() == LOG_DOCK


class TestTheHandlerIsOwnedByTheApplication:
    def test_the_window_attaches_one_and_lets_it_go(self, tmp_path: Path) -> None:
        """A handler left attached to a deleted widget turns the next log line
        into a crash, in a process that is already shutting down."""
        with Nanoscope(settings_path=tmp_path / "settings.json") as container:
            window = MainWindow(container)

            assert _view_handlers() == 1

            window.close()

            assert _view_handlers() == 0

    def test_a_second_window_replaces_rather_than_duplicates(self, tmp_path: Path) -> None:
        """The rule ADR-0051 tagged the handlers for: configuring twice must
        replace, or a restarted window logs everything twice."""
        with Nanoscope(settings_path=tmp_path / "settings.json") as container:
            MainWindow(container)
            MainWindow(container)

            assert _view_handlers() == 1


class TestAHandlerWhoseWindowIsGone:
    """The bug this task turned up, in a test of its own.

    A window that is never closed leaves the handler attached; Qt then deletes
    the C++ object behind it, and the next log line raises `RuntimeError` inside
    `emit` — which `logging.handleError` prints, with a traceback, at whoever is
    watching stderr. It was found from another test file entirely, where it
    turned an unrelated assertion about "no traceback in the output" red.
    """

    def test_it_goes_quiet_instead_of_printing_a_logging_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stream = LogStream()
        handler = stream.handler
        attach_view_log(handler)
        handler._stream = _Deleted()  # type: ignore[attr-defined]

        logger.error("the window is gone")
        logger.error("and so is this one")

        captured = capsys.readouterr()
        assert "Logging error" not in captured.err
        assert _Deleted.attempts == 1, "a dead handler must stop trying, not fail every line"


class _Deleted:
    """What a `QObject` whose C++ half has been deleted does when touched."""

    attempts = 0

    @property
    def logged(self) -> object:
        _Deleted.attempts += 1
        raise RuntimeError("Internal C++ object (LogStream) already deleted.")


def _view_handlers() -> int:
    return len([h for h in logging.getLogger().handlers if h.name == "nanoscope:view"])
