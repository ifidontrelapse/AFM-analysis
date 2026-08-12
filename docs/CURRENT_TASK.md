# CURRENT TASK

**ID:** `M5-T03`
**Title:** One source of colour truth, and a contrast check that can fail
**Milestone:** M5 — GUI shell, third task
**Defect:** — · **ADR:** ADR-0002 decided dark-only; **ADR-0054** records how it is built
**Branch:** `feat/m5-gui-shell`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

ADR-0002: *"Dark theme only. One theme, defined by design tokens plus a QSS stylesheet in
`gui/theme/`. No runtime theme switching, no light variant."* M5-T02 deliberately kept Qt's
defaults so the theme would arrive as one visible change rather than as a diff nobody can see.

It is also the last task before panels start landing (M5-T04, T05, T08). A panel written against
no theme is a panel restyled twice.

---

## The decisions this task has to make

**1. Where does a colour live, and how is "one source of truth" *enforced*?** In
`gui/theme/tokens.py`, and enforced by a **test that fails on a raw hex value in the stylesheet.**

A rule that says "do not hardcode colours" and is checked by review is a rule that lasts until the
first hurried commit. The QSS carries `@token` placeholders, substituted at load; a literal `#1e1e1e`
anywhere in it turns a test red.

**2. QSS alone, or QSS plus a palette?** Both, from the same tokens.

A stylesheet does not reach everything Qt draws — tooltips, dialog buttons, the text cursor, the
disabled states of built-in widgets. `Fusion` plus a `QPalette` built from the tokens gives a
coherent base; QSS refines what the palette cannot express. Two consumers, one table of values.

**3. Is the theme readable?** **Checked, not asserted.** Relative luminance and a contrast ratio are
ten lines of arithmetic, and they turn "the dark theme is fine" from an opinion into a test: every
text-on-background pair in the tokens must clear **4.5:1**, the WCAG AA threshold for body text.

This is the one thing in a theme task worth being strict about. A dark theme that looks good on the
author's monitor and is unreadable on a laboratory projector is the normal outcome, and the only
defence is a number.

**4. Runtime switching?** No — ADR-0002 already decided it, and the tokens being a Python module
rather than a settings key is what keeps it decided.

---

## Scope

**In scope**

1. `gui/theme/tokens.py` — colours, spacing, radii, the type scale; each colour with its purpose
2. `gui/theme/style.qss` — the stylesheet, in `@token` placeholders only
3. `gui/theme/__init__.py` — `apply_theme(app)`: Fusion, the palette, the substituted QSS
4. `MainWindow` and the launcher wired to it
5. **ADR-0054** — tokens as the only source, palette *and* QSS, the contrast floor
6. Tests: every placeholder resolves, **no raw hex in the QSS**, contrast ≥ 4.5:1 for text pairs,
   the theme applies to a real `QApplication`

**Out of scope**

- **A light theme or a switcher** — ADR-0002 said no, twice
- **Icons** — they arrive with the panels that need them
- **Per-widget polish** for panels that do not exist yet

---

## Definition of done

- [x] Tokens, stylesheet and loader, with no colour written twice
- [x] A contrast test that would fail on an unreadable pair
- [x] A test that fails on a raw hex value in the QSS
- [x] ADR-0054
- [x] `make check` green — 888 tests, golden byte-identical
- [x] Docs, the ADR index
- [x] Commit: `M5-T03: one source of colour truth, and a contrast check that can fail`

---

## What it turned up

**The substitution ran over the stylesheet's own comment** — the one explaining what an `@{token}`
is. A checker that reads prose as code fails on the documentation telling it what it does. Comments
are stripped before substitution now, which Qt does not mind because it ignores them anyway.

**`@space_mdpx` parsed as one token name.** Token names and CSS units are both lowercase letters,
so the placeholder needed a delimiter — `@{space_md}px`. Found by writing it the ambiguous way
first and getting a `KeyError` naming a token nobody had defined, which is at least the failure
mode the loader was designed for.

**The contrast floor changed the palette rather than the other way round.** The greens and ambers
are lighter than a first instinct because 4.5:1 said so, which is the entire argument for having a
number instead of an eye.

**Verified rather than assumed:** a wheel was built and inspected, and `style.qss` is in it. A
theme that only works from a checkout is not a theme.
