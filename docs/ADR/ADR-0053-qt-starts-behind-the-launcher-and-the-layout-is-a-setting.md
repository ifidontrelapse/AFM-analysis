# ADR-0053 — Qt starts behind the launcher, and the layout is a setting

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M5-T02)
- **Affects:** `gui/`, `app/main`, dependencies · M5 · every panel task after this one

## Context

M5-T01 left the `--gui` branch with a sentence in it. This task fills it, and in doing so adds the
first dependency the size of Qt and the first line of code in `gui/` — the layer every rule so far
has been written to protect (Architecture §2.3, PROJECT_RULES §2.5/§2.7, and M4-T15's import guard).

W2 has been open since the audit: *"no UI at all."*

## Decision

### 1. `QApplication` is created in `gui/`, and `app/main.py` imports it inside the branch

`gui/launcher.py` owns the `QApplication`, the window and the event loop. `app/main.py` imports it
**inside** `_launch_gui()`.

Not a style preference: M4-T15's guard fails if any module outside `gui/` imports Qt statically, and
the guard is right — the headless entry point is the one CI runs, the one that works on a machine
with no display, and the one an operator uses when a project will not open. Qt loads when a window
is asked for and never otherwise.

### 2. The window holds the container and constructs nothing

`MainWindow(app: Nanoscope)`. No repository, no device manager, no settings store built here.
Opening a project is `Nanoscope.open`, which attaches the log and reads the integrity report as one
act (ADR-0052).

This is the first widget that could break Architecture §2.3, and the shape it sets is the one every
panel after it copies.

`open_project(path)` is separate from `choose_project()` so the *action* can be tested without a
modal dialog — and so a "recent projects" menu has something to call.

### 3. The layout is a setting, not a `QSettings`

`saveState()` / `saveGeometry()` return bytes; they are base64-encoded into the **application**
scope of M4-T10's settings.

`QSettings` would be a second settings system, in a second location, with a second file format, for
one value — while ADR-0047 already built a store with two scopes and a rule for choosing between
them. A window layout follows the **operator**, not the work, so it goes in
`~/.config/nanoscope/settings.json` and never into a project directory.

An unreadable stored layout — one written by an older version — is **ignored with a log line**, not
raised: the answer to a layout Qt cannot parse is the default layout, not a refusal to start.

### 4. Every dock is a placeholder that names the task filling it

Project (M5-T04), Properties (M5-T06), Log (M5-T08), and the central widget (M5-T05). Each says so
on its face, and a test asserts it.

An empty panel is a **promise** when it names its task and a **bug** when it does not. The
alternative — starting each panel here and finishing it badly — is how a shell task becomes four
tasks done at 40%.

Each dock also gets an `objectName`, because a dock without one is silently dropped by
`restoreState()`, which is the kind of bug that appears once, in a user's config, months later.

### 5. Headless tests, no `pytest-qt`

`QT_QPA_PLATFORM=offscreen` set in `tests/gui/conftest.py` **before** Qt is imported — Qt reads it
when the platform plugin loads, and setting it later does nothing — and one session-scoped
`QApplication`, because Qt permits exactly one per process.

`pytest-qt` would be a dependency for a fixture that is six lines. What the tests assert is not
pixels but wiring: that opening a project goes through the container, that a refusal becomes a
message, that the layout round-trips. *A widget that renders correctly and calls the wrong use case
is the failure this project's rules exist to prevent.*

### 6. PySide6 is a runtime dependency **and** a CI one

M5's exit criterion is *"GUI smoke tests pass headless in CI"*, which cannot happen if the `ci`
dependency group excludes Qt. It is added to both, and the `ci` group keeps its purpose — it still
skips the CUDA wheel, which is what it was created for.

## Consequences

**Positive**

- W2 is no longer "no UI at all": `nanoscope --gui` opens a window that opens a project.
- The headless path is untouched and still the one CI runs; Qt is absent from it, by test.
- The layout persists through machinery that already existed, with a rule that already had an
  argument behind it.
- The next four panel tasks each replace one placeholder, and know it.

**Negative**

- A large dependency now installs by default. It is the product's UI toolkit (ADR-0002), and the
  alternative — an optional extra — would make `nanoscope --gui` fail for the operator this
  application is *for*.
- Offscreen Qt is not a screen: it clamps windows to an 800x600 virtual display, which cost one
  confusing test failure before the size in that test was made to fit. Written into the test's
  docstring so the next person does not re-derive it.
- `MainWindow` will grow as panels land. The line to hold is §2: it holds and it displays; it does
  not decide.

**Neutral**

- No theme yet. M5-T03 owns it, and keeping Qt's default here means the theme arrives as one
  visible change rather than as a diff nobody can see.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Create `QApplication` in `app/main.py` | Qt in the headless entry point; M4-T15's guard fails, and correctly |
| Build the container inside `MainWindow` | Exactly what PROJECT_RULES §2.7 forbids, and it makes the window untestable |
| `QSettings` for the layout | A second settings system, location and format, for one value |
| The layout in the project | A window position is about the person, not the work — ADR-0047's rule |
| Refuse to start on an unreadable layout | A stored byte string from an older version stops the application |
| Real panels now instead of placeholders | Four tasks started early and finished badly |
| `pytest-qt` | A dependency for a six-line fixture |
| PySide6 as an optional extra | `nanoscope --gui` fails for the operator the application exists for |

## Compliance

- `tests/gui/test_main_window.py` runs headless and covers the menus, the toolbar, every dock and
  its placeholder text, dock object names, opening a project through the container, the status bar
  carrying the integrity report, a refusal without a traceback, closing, and the layout saved,
  restored and survived when unreadable.
- M4-T15's guard still passes: no module outside `gui/` imports Qt, statically or transitively.
- The real event loop was started under `QT_QPA_PLATFORM=offscreen` and exited 0.

## References

- ADR-0002 (Qt6 / PySide6, dark theme only) — the toolkit decision this implements
- ADR-0052 (the entry point works before there is a window) — the branch this fills
- ADR-0047 (a preference belongs to the operator or to the work) — §3's rule
- ADR-0040 (the repository reports) — the report the status bar shows
- `docs/Architecture.md` §2.3 W2, §3.2 · PROJECT_RULES §2.5, §2.7, §6
