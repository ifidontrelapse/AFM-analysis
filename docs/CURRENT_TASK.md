# CURRENT TASK

**ID:** `M5-T09`
**Title:** A settings dialog that says whose setting it is
**Milestone:** M5 — GUI shell, ninth and last task
**Defect:** — · **ADR:** **ADR-0060**
**Branch:** `feat/m5-gui-shell`
**Status:** **done 2026-08-13.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

It is the last item in M5's list, and it collects an obligation M4-T10 wrote into a docstring:

> `Settings.scope_of(key)` — *"What a settings dialog needs to say **this project overrides your
> default** instead of showing a value with no explanation."*

That method has had no caller since the day it was written. ADR-0047 built two stores and a rule for
choosing between them, and a dialog that writes without saying which one it wrote to is exactly the
failure that ADR predicted: *"guessing it wrong either leaks one project's choice into every other,
or hides a global preference inside one directory."*

There is also a stored preference nothing can set. `DEVICE_SETTING` is read on every
`select_device()` (M5-T01, ADR-0049), and the only way to express it today is to edit
`~/.config/nanoscope/settings.json` by hand.

---

## The decisions this task has to make

**1. What does the dialog edit?** Three things that exist, and nothing that does not.

- **Device** — the preference `Nanoscope.select_device` already reads, offered as *Automatic* plus
  the devices this machine actually has (`DeviceManager.available()`), so the list is a fact about
  the hardware rather than a menu of four names three of which will fail.
- **Default colormap** — what a scan opens in.
- **Log level** — the setting a support conversation asks for first, and the one that has to survive
  a restart to be useful.

A settings dialog is where invented options accumulate; every row here has a reader that existed
before the dialog did.

**2. Which scope does it write?** The operator's, always — and it says so out loud.

Project-scope rows are not offered, because the project-scoped settings this application writes are
**none**: analysis parameters arrive in M6, and that is when the second scope earns a tab. What the
dialog *does* do is read `scope_of` and, where an open project overrides a key, say so beside the
row rather than showing a value the edit will not change. That is M4-T10's sentence, discharged.

**3. When does a change take effect?** Immediately where it can, and the row says when it cannot.

- the log level applies to the running process on OK;
- the colormap becomes the default for the next scan shown — the toolbar combo is *this scan*, the
  dialog is *the default*, which keeps one control from silently overwriting the other;
- the device applies to the next analysis, which in M5 means M6.

**4. Does a stored log level survive a restart?** Yes, and `--debug` still wins.

`app/main.py` reads it before configuring logging; an explicit flag beats a stored preference,
because somebody typing `--debug` is answering the question right now.

**5. How do panels reach a preference?** Through the session, like everything else.

`preference(key, default)` and `remember(key, value)` on `SessionViewModel`, plus a
`settings_changed` signal — panels may not import the composition root (ADR-0057), and a viewer
reaching for `JsonSettings` would be the same hole in a different wall.

---

## Scope

**In scope**

1. `gui/dialogs/settings.py` — the dialog: device, colormap, log level, and the override note
2. `gui/viewmodels/session.py` — `preference`, `remember`, `settings_changed`
3. `gui/panels/viewer.py` — the stored colormap as the default, updated when it changes
4. `MainWindow` — File → *Settings…*
5. `app/main.py` — the stored log level at startup, with `--debug` winning
6. **ADR-0060** — what a settings dialog is allowed to contain, and whose setting it writes
7. Tests: every row round-trips into the store, the device list is the machine's, an overriding
   project is announced, the log level applies immediately and at startup, and the viewer picks up a
   changed default

**Out of scope**

- **Project-scope settings** — nothing writes one yet; M6's analysis parameters are the trigger
- **A reset-to-defaults button** — the store has no notion of a default to reset *to*; deleting a
  key is a different feature with a different question behind it
- **Theme options** — ADR-0002: one dark theme, no switcher

---

## Definition of done

- [x] The dialog reads and writes the three keys, in the operator's scope
- [x] `scope_of` has its first caller, and an overriding project is said out loud
- [x] The log level applies now and after a restart; `--debug` still wins
- [x] ADR-0060 + the ADR index
- [x] `make check` green — 1034 tests, golden byte-identical
- [x] Docs: `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`
- [x] Commit: `M5-T09: a settings dialog that says whose setting it is`

---

## What it turned up

**The dialog showed the project's value in a control that writes the operator's.** Found by opening
it against a project with an override: `preference()` merges project-first, so the combo read
`bone` from the project while OK would have written `bone` as the operator's default — **ADR-0047's
first failure mode, in one screen**: one project's choice promoted to every project. The control now
reads `own_preference` (the application store alone) and the note explains why the effective value
differs. The note existed before the bug did and hid it: the screen looked right.

**The device key was defined in `app/container.py`, where the dialog could only retype it.** Moved
to `application/settings.py` beside the other two, because a settings key typed twice is a
preference that silently does nothing on one side of the application — the one class of typo that
cannot fail loudly.

**`main` configured logging before constructing the container**, so a stored level could not be
read at all. Reordered, with a comment: nothing logs before that line, and if anything ever does the
ordering has to be revisited.
