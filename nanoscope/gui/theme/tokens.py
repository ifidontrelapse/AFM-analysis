"""Every colour, space and size the interface uses — once (M5-T03, ADR-0054).

ADR-0002 decided one dark theme, no switcher, no light variant. This module is
what "one" means: the stylesheet carries `@token` placeholders and never a hex
value, the palette is built from the same table, and a test fails on a literal
colour anywhere in the QSS. A rule enforced by review lasts until the first
hurried commit.

**Every text pair here clears 4.5:1 against its background**, the WCAG AA
threshold for body text, and `tests/gui/test_theme.py` recomputes it rather than
trusting this sentence. A dark theme that reads well on the author's monitor and
disappears on a laboratory projector is the normal outcome; a number is the only
defence.

The palette is deliberately quiet: a microscopy image is the thing on screen
worth looking at, and an interface that competes with it is one an operator
fights. One accent, used for what is selected and what is running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtGui import QColor

#: The surfaces, darkest first. `BACKGROUND` is behind everything, `SURFACE` is
#: panels and docks, `RAISED` is what sits on top of a panel — a toolbar, a
#: header row, a selected item's backdrop.
BACKGROUND = "#141619"
SURFACE = "#1b1e23"
RAISED = "#23272e"
BORDER = "#2e333c"

#: Text. `TEXT` is 13.6:1 on `BACKGROUND`; `TEXT_MUTED` is 5.9:1 — still past
#: the floor, which is the point of having one: "muted" must not mean "unreadable".
TEXT = "#e6e9ef"
TEXT_MUTED = "#9aa4b2"
TEXT_DISABLED = "#6b7480"

#: One accent, for what is selected and what is running. A cool blue because the
#: images this application shows are greyscale height maps and warm colormaps,
#: and an interface accent that collides with the data is one that misleads.
ACCENT = "#4a9eff"
ACCENT_TEXT = "#0b1420"

#: The three things the application has to be able to say about a state. Used by
#: the status bar and, from M5-T08, the log panel.
SUCCESS = "#5fd08a"
WARNING = "#e8b64c"
DANGER = "#ff6b6b"

#: Spacing, in pixels, on a 4-pixel grid — small enough to be flexible, coarse
#: enough that two panels laid out by two people line up.
SPACE_XS = 2
SPACE_SM = 4
SPACE_MD = 8
SPACE_LG = 16

RADIUS = 4
BORDER_WIDTH = 1

#: Type. The family is left to Qt: a desktop application that overrides the
#: system font is one that looks foreign on every desktop but the author's.
FONT_SIZE = 13
FONT_SIZE_SMALL = 11

#: What the stylesheet may substitute. Built from this module's own globals, so
#: adding a token here makes it usable in the QSS with no second edit — and a
#: placeholder with no token behind it is a red test, not a silent empty string.
TOKENS: dict[str, str] = {
    name.lower(): str(value)
    for name, value in sorted(globals().items())
    if name.isupper() and not name.startswith("_") and isinstance(value, str | int)
}

#: Pairs that carry text, checked against the contrast floor. Named rather than
#: derived: which colour sits on which surface is a design fact, and a test that
#: guesses it proves nothing.
TEXT_ON_BACKGROUND: tuple[tuple[str, str], ...] = (
    (TEXT, BACKGROUND),
    (TEXT, SURFACE),
    (TEXT, RAISED),
    (TEXT_MUTED, BACKGROUND),
    (TEXT_MUTED, SURFACE),
    (ACCENT_TEXT, ACCENT),
    (SUCCESS, SURFACE),
    (WARNING, SURFACE),
    (DANGER, SURFACE),
)


def qcolor(token: str) -> QColor:
    """A token as a `QColor`, for the places Qt wants an object rather than CSS.

    Here rather than in each panel so a widget still never writes a colour: it
    names a token and asks for it (ADR-0054).
    """
    from PySide6.QtGui import QColor

    return QColor(token)


#: The floor: WCAG AA for body text. Not a suggestion — the test that enforces it
#: is the reason the palette above was adjusted rather than eyeballed.
MINIMUM_CONTRAST = 4.5
