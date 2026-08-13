# ADR-0060 — A setting is offered only if something reads it

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** operator + agent (M5-T09)
- **Affects:** `gui/dialogs`, `gui/viewmodels`, `application/settings`, `app/main` · M5 · M6

## Context

M4-T10 built two settings stores and a rule for choosing between them (ADR-0047), and wrote a method
whose docstring named the task that would call it:

> `Settings.scope_of(key)` — *"What a settings dialog needs to say **this project overrides your
> default** instead of showing a value with no explanation."*

It has had no caller since. Meanwhile `DEVICE_SETTING` is read on every `select_device()`
(ADR-0049) and the only way to express it has been to edit `~/.config/nanoscope/settings.json` by
hand.

A settings dialog is also the place where invented options accumulate: it is the one screen where
adding a control feels free, and where every control added is a promise that something reads it.

## Decision

### 1. Three rows, each with a reader that predates the dialog

- **Device** — resolved by `Nanoscope.select_device` (ADR-0049);
- **Default colormap** — read by the viewer (M5-T05);
- **Log level** — read by `app/main.py` at startup, and applied to the running process on OK.

Nothing else is offered. The rule is testable, which is the point: for each row, a test asserts the
value reaches the store the existing reader already looks in.

### 2. It writes the operator's scope, and says so

One line under the form: *"These are your preferences, and follow you to every project."*

Project-scoped rows are not offered, because this application writes **no** project-scoped setting
yet — analysis parameters arrive with M6, and that is when the second scope earns a tab. ADR-0047's
warning is about a caller that *guesses* between the two, and the answer here is to name the one
being written rather than to build a scope selector nothing would populate.

### 3. The control shows the operator's own value, and an override is said out loud

`own_preference` reads the application store directly; `preference` (merged, project first) is what
panels use to *behave*.

Showing the merged value in a control that writes the application scope is ADR-0047's first failure
mode in one screen: the operator sees a project's answer, presses OK, and copies that project's
choice into every other project they open. So the control shows their own value, and where the open
project overrides the key, a line beneath it says the project's value wins — which is `scope_of`'s
first caller, six milestones after it was written.

### 4. A change takes effect where it can, and the row says where it cannot

The log level applies to the running process immediately, because an operator setting DEBUG is about
to reproduce something. The colormap is the default for the **next** scan shown. The device applies
to the next analysis, which in M5 means M6.

**The toolbar's colormap combo is *this scan*; the dialog is *the default*.** One control reads the
key and the other writes it — two controls writing one key is a fight the operator loses either way.

### 5. The stored log level survives a restart, and `--debug` beats it

`app/main.py` constructs the container first, reads the preference, then configures logging.
Somebody typing `--debug` is answering the question right now; a stored preference is an answer they
gave once.

An unreadable stored value is ignored with a warning rather than being fatal — the same rule the
device preference already follows (ADR-0052), and for the same reason: a typo in a settings file
must not stop the application starting.

### 6. The keys live in `application/settings.py`

`DEVICE_SETTING`, `COLORMAP_SETTING`, `LOG_LEVEL_SETTING`. A settings key is a string, and a string
typed twice is a preference that silently does nothing on one side of the application — the argument
PROJECT_RULES §3 makes about magic constants, at the one place where getting it wrong cannot fail
loudly. `app/container.py` re-exports the device key it used to define.

### 7. Panels reach preferences through the session

`preference`, `own_preference`, `remember`, and a `settings_changed` signal. Panels may not import
the composition root (ADR-0057), and a viewer reaching for `JsonSettings` would be the same hole in
a different wall.

## Consequences

**Positive**

- The device preference is reachable without a text editor, and the list is the machine's own
  hardware rather than four enum names.
- `scope_of` has a caller, and the dialog cannot silently promote a project's choice to a global
  default.
- A support conversation can begin "set the log level to Debug and reproduce it" and end with a file
  that already has the lines.

**Negative**

- No *reset to defaults*. The store has no notion of a default to reset to, and deleting a key is a
  different feature with a different question behind it ("whose default?").
- The dialog is modal and applies on OK. A live-preview settings screen would need every consumer to
  cope with a value that changes mid-edit, for three rows.
- A project override can only be *seen* here, not edited. The operator is told the project wins and
  given no way to change that from this screen — deliberate while nothing writes a project setting,
  and a gap the moment M6 does.

**Neutral**

- The log level is stored as an `int` — `logging`'s own numbers, so `logging.getLogger().setLevel`
  takes it unchanged, and a stored `20` is legible to anyone who has read the module once.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A scope selector on every row | Nothing writes a project setting yet; a control with one usable option |
| Show the merged value in the control | ADR-0047's first failure mode: OK copies a project's choice into every project |
| Let the viewer's combo write the preference too | Two controls writing one key, and a feedback loop between them |
| A live-apply settings screen | Three rows do not justify every consumer handling mid-edit changes |
| Offer every `logging` level | Nobody wants `CRITICAL`-only; a level below `DEBUG` is a number, not a choice |
| Keep the device key in `app/container.py` | The dialog would type the string a second time, where a typo fails silently |
| A "reset to defaults" button | The store has no defaults; the question behind the button is unanswered |

## Compliance

- `tests/gui/test_settings_dialog.py` asserts each row round-trips into the store its reader uses,
  that the device list is `DeviceManager.available()`, that a stored value this build does not offer
  is ignored rather than fatal, that cancelling writes nothing, that the log level applies to the
  running process, and that an overriding project is announced while the control still shows the
  operator's own value.
- `tests/integration/test_entry_point.py::TestTheStoredLogLevel` covers §5, including `--debug`
  winning and an unreadable value being ignored.
- `tests/gui/test_settings_dialog.py::TestTheViewerFollowsTheDefault` pins §4's split between the
  toolbar and the dialog.

## References

- ADR-0047 (a preference belongs to the operator or to the work) — the two stores, and the failure
  mode §3 avoids
- ADR-0049 (no torch is a CPU, and ROCm is not CUDA) — the device list §1 offers
- ADR-0051 — the log the level in §1 governs
- ADR-0057 §2 — why panels reach settings through the session
- `Settings.scope_of`, M4-T10 — the docstring this task was written against
