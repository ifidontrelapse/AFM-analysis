# CURRENT TASK

**ID:** `M5-T02`
**Title:** A window that opens a project, and a layout that remembers itself
**Milestone:** M5 — GUI shell, second task
**Defect:** W2 (no UI at all) · **ADR:** **ADR-0053**
**Branch:** `feat/m5-gui-shell`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M5-T01 left a branch with a sentence in it: `--gui` says the window arrives in M5-T02. This is
M5-T02. It is also the first task in the project's history to add a dependency the size of Qt, and
the first line of code in `gui/` — the layer every rule so far has been written to protect.

---

## The decisions this task has to make

**1. Where does `QApplication` get created?** In `gui/`, not in `app/`.

`app/main.py` must not import PySide6, or M4-T15's guard fails — and the guard is right: the
headless entry point is the one CI runs. So `gui/launcher.py` owns `QApplication`, and `main.py`
imports it **inside** the `--gui` branch. Qt is loaded when a window is asked for and never
otherwise.

**2. What does the window *hold*?** The container, and nothing of its own.

`MainWindow(QMainWindow)` takes a `Nanoscope` and calls it. No repository, no device manager, no
settings store constructed here — Architecture §2.3 and PROJECT_RULES §2.7 both say so, and this is
the first widget that could break either.

**3. Where does the layout live?** In the **application** scope of M4-T10's settings, base64
encoded.

Qt's `saveState()`/`saveGeometry()` return a `QByteArray`, and there is already a settings store
with two scopes and a rule for choosing between them (ADR-0047): a window layout follows the
**operator**, not the work, so it goes in `~/.config/nanoscope/settings.json` and not into the
project. `QSettings` is deliberately not used — a second settings system, in a second location,
with a second file format, for one value.

**4. What do the docks contain today?** Placeholders that say which task fills them.

The project explorer is M5-T04, the image viewer M5-T05, the log panel M5-T08. A dock with a label
naming its task is honest; a dock with a half-built panel in it is M5-T04 started early and finished
badly.

**5. How is a window tested without a screen?** `QT_QPA_PLATFORM=offscreen`, which PROJECT_RULES §6
already requires. No `pytest-qt`: one fixture creating a `QApplication` is what this needs, and a
dependency for a fixture is a dependency for nothing.

**6. Which of M4's obligations can be paid here?** Neither, and both get closer.
`remove_image`'s confirmation (ADR-0044) needs the project explorer (M5-T04); the job listener's
thread hop (ADR-0043) needs the job runner integration (M5-T07). They are named again here so they
stay attached to the tasks that own them.

---

## Scope

**In scope**

1. PySide6 as a dependency, and in the `ci` group — M5's criterion is *"GUI smoke tests pass
   headless in CI"*, which needs it there
2. `gui/main_window.py` — menus, a toolbar, four dockable panels, a status bar
3. `gui/launcher.py` — `QApplication`, the window, the event loop; `--gui` wired to it
4. Layout persistence through the application settings
5. **ADR-0053** — Qt stays behind the launcher, the layout is a setting not a `QSettings`,
   placeholders name their task
6. Headless tests: the window builds, opens a project, reports a refusal without a traceback,
   saves and restores its layout

**Out of scope**

- **The panels themselves** — M5-T04, T05, T08, T09
- **The theme** — M5-T03. This task uses Qt's defaults so the theme lands as one visible change
- **Running analyses from the window** — M5-T06/T07

---

## Definition of done

- [x] `nanoscope --gui` opens a window with menus, docks and a status bar
- [x] The window holds the container and constructs nothing
- [x] Layout saved and restored through M4-T10's settings
- [x] ADR-0053
- [x] Headless tests, and the Qt guard still green
- [x] `make check` green — 867 tests, golden byte-identical
- [x] Docs, the ADR index, `Roadmap.md`
- [x] Commit: `M5-T02: a window that holds the container and nothing else`

---

## What it turned up

**An existing test hung instead of failing.** M5-T01's `--gui` test called the flag directly,
because the branch printed a sentence and returned. When this task made it launch a real event
loop, the test stopped returning — and **a hang is worse than a failure**, because there is no
timeout in the suite to turn it back into one. The entry-point test now asserts the *handover* to
the launcher, and a new `tests/gui/test_launcher.py` enters the loop deliberately with a timer that
knows how to leave it.

**`from conftest import revert_to` was ambiguous the moment a second conftest existed.** Adding
`tests/gui/conftest.py` made the bare module name resolve to whichever directory pytest had put on
`sys.path` first, and three integration files failed to import. Fixtures belong in a conftest;
**importable helpers belong in a module with a name of their own** — now
`tests/integration/schema_history.py`.

**The whole-layer test's `"PySide6" not in sys.modules` was a weaker copy of a guard that already
exists**, and it broke for exactly the in-process reason `test_ports.py` was repaired for one task
earlier. Deleted, with a comment pointing at the subprocess check that does it properly.

**Offscreen Qt is not a screen:** it clamps windows to an 800x600 virtual display, which cost one
confusing failure until the layout test asked for a size that fits.
