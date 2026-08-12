"""Applying the one theme (M5-T03, ADR-0054).

ADR-0002: dark only, no switcher. `apply_theme(app)` does three things, all from
`tokens.py`:

1. **Fusion**, because the native styles on some desktops ignore half a
   stylesheet and produce a window that is dark in places;
2. a **`QPalette`**, because a stylesheet does not reach everything Qt draws —
   tooltips, dialog buttons, disabled states, the text cursor;
3. the **stylesheet**, with `@token` placeholders resolved.

Two consumers, one table of values. A colour written twice is a colour that will
disagree with itself.
"""

from __future__ import annotations

import re
from importlib import resources

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from nanoscope.gui.theme import tokens

STYLESHEET_NAME = "style.qss"

#: `@{name}`, braces and all. The delimiter is not decoration: token names and
#: CSS units are both lowercase letters, so `@space_mdpx` reads as one name and
#: resolves to nothing — found by writing it that way first.
_PLACEHOLDER = re.compile(r"@\{([a-z_]+)\}")

#: `/* … */`. Stripped **before** substitution: this file's own header explains
#: what an `@token` is, and a checker that reads prose as code fails on the
#: documentation telling it what it does. Qt ignores comments anyway.
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def stylesheet() -> str:
    """The QSS with every `@token` resolved.

    Raises:
        KeyError: a placeholder with no token behind it. Loud, because the
            alternative is a widget silently styled with an empty string —
            which looks like a design decision and is a typo.
    """
    template = resources.files(__package__).joinpath(STYLESHEET_NAME).read_text(encoding="utf-8")
    return _PLACEHOLDER.sub(_resolve, _COMMENT.sub("", template)).strip()


def _resolve(match: re.Match[str]) -> str:
    name = match.group(1)
    if name not in tokens.TOKENS:
        raise KeyError(f"{STYLESHEET_NAME} uses @{name}, which is not a token in tokens.py")
    return tokens.TOKENS[name]


def palette() -> QPalette:
    """The base colours, for everything a stylesheet cannot reach."""
    colours = QPalette()
    background, surface, raised = (
        QColor(tokens.BACKGROUND),
        QColor(tokens.SURFACE),
        QColor(tokens.RAISED),
    )
    text, muted, disabled = (
        QColor(tokens.TEXT),
        QColor(tokens.TEXT_MUTED),
        QColor(tokens.TEXT_DISABLED),
    )

    for role, colour in (
        (QPalette.ColorRole.Window, background),
        (QPalette.ColorRole.Base, surface),
        (QPalette.ColorRole.AlternateBase, raised),
        (QPalette.ColorRole.Button, raised),
        (QPalette.ColorRole.ToolTipBase, raised),
        (QPalette.ColorRole.WindowText, text),
        (QPalette.ColorRole.Text, text),
        (QPalette.ColorRole.ButtonText, text),
        (QPalette.ColorRole.ToolTipText, text),
        (QPalette.ColorRole.PlaceholderText, muted),
        (QPalette.ColorRole.Highlight, QColor(tokens.ACCENT)),
        (QPalette.ColorRole.HighlightedText, QColor(tokens.ACCENT_TEXT)),
        (QPalette.ColorRole.Link, QColor(tokens.ACCENT)),
    ):
        colours.setColor(role, colour)

    #: Disabled text is a state Qt draws itself, and leaving it at the default
    #: puts near-black text on a dark surface — legible in the designer's head
    #: and nowhere else.
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        colours.setColor(QPalette.ColorGroup.Disabled, role, disabled)

    return colours


def apply_theme(app: QApplication) -> None:
    """Dress the application. Called once, by the launcher."""
    app.setStyle("Fusion")
    app.setPalette(palette())
    app.setStyleSheet(stylesheet())
