"""The theme, checked rather than admired (M5-T03, ADR-0054).

Three things are enforced here, and the middle one is the reason the file
exists.

1. **Every placeholder resolves.** A `@{token}` with nothing behind it would
   style a widget with an empty string, which looks like a design decision.
2. **Contrast.** Every text pair clears 4.5:1 — WCAG AA for body text —
   recomputed from the tokens rather than trusted. A dark theme that reads well
   on the author's monitor and disappears on a laboratory projector is the
   normal outcome, and a number is the only defence.
3. **No colour is written twice.** A literal hex in the stylesheet fails, which
   is what "one source of colour truth" has to mean to survive a hurried commit.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import pytest
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from nanoscope.gui.theme import STYLESHEET_NAME, apply_theme, palette, stylesheet, tokens

pytestmark = pytest.mark.usefixtures("qt_app")

HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def raw_stylesheet() -> str:
    return resources.files("nanoscope.gui.theme").joinpath(STYLESHEET_NAME).read_text("utf-8")


def relative_luminance(colour: str) -> float:
    """WCAG 2.1's relative luminance of an sRGB colour.

    Written out rather than imported: it is six lines, and the alternative is a
    dependency whose only job is to be trusted about arithmetic that can be
    checked against the specification in a minute.
    """
    channels = []
    for component in QColor(colour).getRgb()[:3]:
        c = component / 255
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


class TestTheContrastFloor:
    def test_the_measure_is_right_before_it_is_used(self) -> None:
        """A check that cannot fail is decoration. Black on white is 21:1 and
        white on white is 1:1 — if these two are wrong, nothing below means
        anything."""
        assert contrast("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
        assert contrast("#ffffff", "#ffffff") == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize(
        ("foreground", "background"),
        tokens.TEXT_ON_BACKGROUND,
        ids=lambda pair: str(pair),
    )
    def test_every_text_pair_is_readable(self, foreground: str, background: str) -> None:
        ratio = contrast(foreground, background)

        assert ratio >= tokens.MINIMUM_CONTRAST, (
            f"{foreground} on {background} is {ratio:.2f}:1, below the "
            f"{tokens.MINIMUM_CONTRAST}:1 floor (WCAG AA for body text)"
        )

    def test_muted_text_is_muted_and_still_readable(self) -> None:
        """ "Muted" must not quietly come to mean "unreadable" — the whole reason
        the floor applies to it too."""
        assert contrast(tokens.TEXT_MUTED, tokens.SURFACE) < contrast(tokens.TEXT, tokens.SURFACE)
        assert contrast(tokens.TEXT_MUTED, tokens.SURFACE) >= tokens.MINIMUM_CONTRAST


class TestOneSourceOfColourTruth:
    def test_the_stylesheet_contains_no_colour_of_its_own(self) -> None:
        """The rule, enforced. A literal `#1e1e1e` here is a colour that will
        one day disagree with the token it was copied from."""
        offenders = HEX.findall(raw_stylesheet())

        assert not offenders, f"{STYLESHEET_NAME} hardcodes {offenders}; use @{{token}} instead"

    def test_every_placeholder_has_a_token(self) -> None:
        """`stylesheet()` raises rather than substituting an empty string, so
        this passes by not raising — and names what it would have said."""
        assert "@{" not in stylesheet()

    def test_an_unknown_placeholder_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard that makes the test above worth having."""
        from nanoscope.gui import theme

        monkeypatch.setattr(theme, "_COMMENT", re.compile(r"$^"))
        monkeypatch.setattr(
            theme.resources,
            "files",
            lambda _: _FakeResource("QWidget { color: @{no_such_token}; }"),
        )

        with pytest.raises(KeyError, match="no_such_token"):
            theme.stylesheet()

    def test_the_tokens_reach_the_stylesheet(self) -> None:
        resolved = stylesheet()

        assert tokens.BACKGROUND in resolved
        assert tokens.ACCENT in resolved
        assert f"{tokens.FONT_SIZE}px" in resolved


class TestApplyingIt:
    def test_the_palette_is_built_from_the_tokens(self) -> None:
        colours = palette()

        assert colours.color(QPalette.ColorRole.Window).name() == tokens.BACKGROUND
        assert colours.color(QPalette.ColorRole.WindowText).name() == tokens.TEXT
        assert colours.color(QPalette.ColorRole.Highlight).name() == tokens.ACCENT

    def test_disabled_text_is_not_left_to_qt(self) -> None:
        """Qt's default disabled colour on a dark palette is near-black on dark
        — legible in the designer's head and nowhere else."""
        colours = palette()

        disabled = colours.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText)
        assert disabled.name() == tokens.TEXT_DISABLED

    def test_applying_it_dresses_the_application(self) -> None:
        app = QApplication.instance()
        assert app is not None

        apply_theme(app)

        assert app.styleSheet().startswith("QWidget")
        assert app.palette().color(QPalette.ColorRole.Window).name() == tokens.BACKGROUND

    def test_the_stylesheet_ships_with_the_package(self, tmp_path: Path) -> None:
        """It is package data, not a file beside the source tree: an installed
        wheel has no `nanoscope/gui/theme/style.qss` unless the build includes
        it, and a theme that only works from a checkout is not a theme."""
        assert resources.files("nanoscope.gui.theme").joinpath(STYLESHEET_NAME).is_file()


class _FakeResource:
    """Enough of `importlib.resources`' interface for one test."""

    def __init__(self, text: str) -> None:
        self._text = text

    def joinpath(self, _name: str) -> _FakeResource:
        return self

    def read_text(self, *_args: object, **_kwargs: object) -> str:
        return self._text
