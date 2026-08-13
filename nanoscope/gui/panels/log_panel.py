"""What the application has been saying, where an operator can read it (M5-T08).

The dock M5-T02 left a placeholder in, and the half of ADR-0051 nobody could
reach: the log went to two rotating JSONL files, and an operator who wanted to
know why an import refused a scan had to find `$XDG_STATE_HOME` in a file
manager.

The file stays authoritative — this is **this session**, bounded, in a
`QPlainTextEdit` whose `maximumBlockCount` is Qt's own ring buffer. A panel that
keeps every line of a day's work is a memory leak with a scrollbar.
"""

from __future__ import annotations

from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels.log_stream import LogLine, LogStream

#: How many lines are kept. Enough to cover an import and its aftermath, few
#: enough that nothing here grows without limit — the file has the rest.
MAX_LINES = 2_000

#: What each level looks like. A warning that looks like every other line is a
#: warning nobody sees; the colours are tokens, because a widget never writes one
#: (ADR-0054).
LEVEL_COLOURS: dict[str, str] = {
    "DEBUG": tokens.TEXT_DISABLED,
    "INFO": tokens.TEXT_MUTED,
    "WARNING": tokens.WARNING,
    "ERROR": tokens.DANGER,
    "CRITICAL": tokens.DANGER,
}


class LogPanel(QWidget):
    """The session's log, newest at the bottom."""

    def __init__(self, stream: LogStream, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.view = QPlainTextEdit(self)
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(MAX_LINES)
        #: A log is columns of fixed-width text or it is a paragraph.
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        stream.logged.connect(self.append)

    def append(self, line: LogLine) -> None:
        """Add one line. Called on the main thread — the signal is queued."""
        colour = LEVEL_COLOURS.get(line.level_name, tokens.TEXT)
        self.view.appendHtml(f'<pre style="color:{colour}; margin:0">{_escaped(line)}</pre>')

    @property
    def text(self) -> str:
        return self.view.toPlainText()


def _escaped(line: LogLine) -> str:
    """A log line is data, and this one is going into HTML.

    An exception message containing `<` would otherwise disappear into the
    markup — which is the log line most worth reading, since somebody wrote a
    repr into it.
    """
    return line.as_text().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
