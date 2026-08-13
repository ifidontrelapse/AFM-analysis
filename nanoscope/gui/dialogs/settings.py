"""The preferences an operator can state, and whose they are (M5-T09, ADR-0060).

Three rows, and every one of them has a reader that existed before this dialog
did: the device `Nanoscope.select_device` resolves (ADR-0049), the colormap a
scan opens in (M5-T05), and the log level a support conversation asks for first
(ADR-0051). A settings dialog is where invented options accumulate, and the rule
that keeps this one honest is that nothing is offered which nothing reads.

**It writes the operator's scope, always.** ADR-0047 built two stores and warned
what happens to a caller that guesses between them — *"either leaks one project's
choice into every other, or hides a global preference inside one directory"*. The
project scope is not offered because this application writes no project-scoped
setting yet; what the dialog does instead is **say when an open project overrides
a key**, which is the sentence `Settings.scope_of` was written for in M4-T10 and
has been waiting for a caller ever since.
"""

from __future__ import annotations

import logging

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nanoscope.application.settings import (
    COLORMAP_SETTING,
    DEVICE_SETTING,
    LOG_LEVEL_SETTING,
)
from nanoscope.application.use_cases.display import COLORMAPS
from nanoscope.gui.theme import tokens
from nanoscope.gui.viewmodels import SessionViewModel

#: What "no preference" looks like in the device list. Stored as `None`, which is
#: what `select_device` reads as *let the policy decide* — the common case, and
#: the one that keeps working when the operator changes machines.
AUTOMATIC = "Automatic (best available)"

#: The levels worth offering. Not every level `logging` defines: nobody wants
#: `CRITICAL`-only, and a level below `DEBUG` is a number, not a choice.
LEVELS: tuple[tuple[str, int], ...] = (
    ("Debug (everything)", logging.DEBUG),
    ("Info (the default)", logging.INFO),
    ("Warning (problems only)", logging.WARNING),
)


class SettingsDialog(QDialog):
    """Ask, store, and apply what can be applied now."""

    def __init__(self, session: SessionViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("Settings")

        self.device = QComboBox(self)
        self.device.addItem(AUTOMATIC, None)
        for device in session.devices():
            #: The devices this machine has, not the four the enum names —
            #: three of which would fail on any given computer (ADR-0049).
            self.device.addItem(f"{device.name} ({device.kind})", str(device.kind))
        _select(self.device, session.own_preference(DEVICE_SETTING))

        self.colormap = QComboBox(self)
        self.colormap.addItems(COLORMAPS)
        self.colormap.setCurrentText(str(session.own_preference(COLORMAP_SETTING, COLORMAPS[0])))

        self.level = QComboBox(self)
        for label, value in LEVELS:
            self.level.addItem(label, value)
        _select(self.level, session.own_preference(LOG_LEVEL_SETTING, logging.INFO))

        form = QFormLayout()
        for label, widget, key in (
            ("Device:", self.device, DEVICE_SETTING),
            ("Default colormap:", self.colormap, COLORMAP_SETTING),
            ("Log level:", self.level, LOG_LEVEL_SETTING),
        ):
            form.addRow(label, widget)
            note = _override_note(session, key)
            if note is not None:
                form.addRow("", note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)

        scope = QLabel("These are your preferences, and follow you to every project.", self)
        scope.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(scope)
        layout.addWidget(buttons)

    def apply(self) -> None:
        """Store the three, apply the one that can take effect now, and close.

        The log level applies to the running process; the colormap is the
        default for the **next** scan shown, which is what keeps the toolbar's
        combo (*this scan*) from fighting this one (*the default*); the device
        applies to the next analysis, which is M6.
        """
        self._session.remember(DEVICE_SETTING, self.device.currentData())
        self._session.remember(COLORMAP_SETTING, self.colormap.currentText())

        level = int(self.level.currentData())
        self._session.remember(LOG_LEVEL_SETTING, level)
        #: Now, not at the next start: an operator setting DEBUG is about to
        #: reproduce something.
        logging.getLogger().setLevel(level)
        self.accept()


def _override_note(session: SessionViewModel, key: str) -> QLabel | None:
    """The sentence M4-T10 asked for, or nothing when there is nothing to say.

    It matters *because* the control above it shows the operator's own value:
    without the note, a project override makes the dialog look wrong.
    """
    if not session.overridden_by_project(key):
        return None
    note = QLabel("This project overrides your default; the project's value wins.")
    note.setStyleSheet(f"color: {tokens.WARNING};")
    note.setWordWrap(True)
    return note


def _select(combo: QComboBox, stored: object) -> None:
    """Put the combo on the stored value, or leave it on its first entry.

    A stored value this version does not offer — a GPU that has been unplugged,
    a level from a newer build — selects nothing rather than raising: a settings
    file describing another machine must not stop the dialog opening.
    """
    index = combo.findData(stored)
    if index >= 0:
        combo.setCurrentIndex(index)
