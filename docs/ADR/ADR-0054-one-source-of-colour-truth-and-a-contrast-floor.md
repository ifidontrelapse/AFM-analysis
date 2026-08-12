# ADR-0054 — One source of colour truth, and a contrast floor that can fail

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M5-T03)
- **Affects:** `gui/theme` · M5 · every panel from M5-T04 onward

## Context

ADR-0002 decided the theme three months ago: *"Dark theme only. One theme, defined by design tokens
plus a QSS stylesheet in `gui/theme/`. No runtime theme switching, no light variant."*

M5-T02 deliberately kept Qt's defaults so the theme would arrive as one visible change. It is also
the last task before panels start landing — and a panel written against no theme is a panel
restyled twice.

## Decision

### 1. Colours live in `tokens.py`, and the rule is enforced by a test

The stylesheet carries `@{token}` placeholders and **no colour of its own**. A literal hex value
anywhere in `style.qss` fails `test_the_stylesheet_contains_no_colour_of_its_own`.

"Do not hardcode colours" checked by review is a rule that lasts until the first hurried commit.
The same argument as ADR-0044's `CHECK` constraints and M4-T02's schema guards: **the rule and its
enforcement ship together, or only the rule does.**

A placeholder with no token behind it raises rather than substituting an empty string — an
unstyled widget looks like a design decision, and this one would be a typo.

### 2. Both a palette and a stylesheet, from the same table

`Fusion`, a `QPalette`, then the QSS. A stylesheet does not reach everything Qt draws — tooltips,
dialog buttons, the text cursor, the disabled state of built-in widgets — and the native styles on
some desktops ignore half a stylesheet, producing a window that is dark in places.

Two consumers, one table of values. **Qt's default disabled colour on a dark palette is near-black
on dark**, which is legible in the designer's head and nowhere else, so `TEXT_DISABLED` is set
explicitly for the disabled group.

### 3. Contrast is a floor, not an opinion

Every text pair in `TEXT_ON_BACKGROUND` must clear **4.5:1**, WCAG AA for body text, recomputed by
the test from relative luminance rather than asserted in a comment.

This is the one part of a theme worth being strict about. A dark palette that reads well on the
author's monitor and disappears on a laboratory projector is the *normal* outcome, and a number is
the only defence. The rule applies to `TEXT_MUTED` too, because **"muted" must not quietly come to
mean "unreadable"**.

The luminance formula is written out rather than depended upon: it is six lines, and the
alternative is a package whose only job is to be trusted about arithmetic anyone can check against
the specification in a minute. A test asserts the *measure* first — black on white is 21:1, white
on white is 1:1 — because a contrast check that cannot fail is decoration.

### 4. The palette is quiet, and that is a design decision

One accent, cool blue, used for what is selected and what is running. The thing on screen worth
looking at is a microscopy image; an interface that competes with it is one an operator fights, and
a warm accent beside a warm colormap is one that misleads.

### 5. No switcher, and the tokens being code is what keeps it that way

ADR-0002 said dark only. Tokens as a Python module rather than a settings key means adding a light
theme is a decision somebody has to make on purpose, not a key somebody sets.

## Consequences

**Positive**

- One place to change a colour, and a test that fails when somebody changes it in two.
- The theme is *legible by measurement*, and the measurement is in the suite.
- The stylesheet ships in the wheel (verified by building one), so the theme works from an install
  and not only from a checkout.
- The panels from M5-T04 onward inherit a styled base and are written once.

**Negative**

- The QSS is not valid CSS until substituted, so an editor cannot preview it. Accepted: the
  alternative is hex values in two files.
- The contrast floor constrains the palette. That is the point, and it is why the greens and ambers
  are lighter than a designer's first instinct.
- Fusion overrides the platform style, so nanoscope will not look like a GTK or KDE application.
  ADR-0002 chose one theme deliberately; consistency across desktops is what that buys.

**Neutral**

- Only widgets that exist are styled. Styling a widget before it exists is styling a guess.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Hex values directly in the QSS | Two places to change a colour, and no way to notice they disagree |
| Generate the QSS from Python strings | The stylesheet stops being a stylesheet anyone can read |
| A colour library for contrast | A dependency to be trusted about six lines of arithmetic |
| Trust the palette by eye | The normal outcome is a theme that fails on somebody else's screen |
| Keep Qt's native style per platform | Half a stylesheet ignored, and a window that is dark in places |
| A light theme "for later" | ADR-0002 said no; a switcher is a second theme to keep correct forever |

## Compliance

- `tests/gui/test_theme.py` enforces: no hex in the stylesheet, every placeholder resolved, an
  unknown placeholder refused, the contrast floor for every declared text pair, the measure itself,
  the palette built from tokens, disabled text set explicitly, and the stylesheet present as
  package data.
- `uv build --wheel` was run and the wheel inspected: `nanoscope/gui/theme/style.qss` is in it.
- No module outside `gui/` imports the theme, and no widget defines a colour of its own.

## References

- ADR-0002 (Qt6 / PySide6, dark theme only) — the decision this implements
- ADR-0053 (Qt starts behind the launcher) — where `apply_theme` is called
- ADR-0044 / M4-T02 — the same "ship the rule with its enforcement" argument, in SQL
- WCAG 2.1 §1.4.3 — the 4.5:1 floor for body text
