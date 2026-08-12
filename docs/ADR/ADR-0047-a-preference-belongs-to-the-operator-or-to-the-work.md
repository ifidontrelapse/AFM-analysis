# ADR-0047 — A preference belongs either to the operator or to the work

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T10)
- **Affects:** `core/ports`, `infrastructure/storage`, `application/settings`, schema v4 · M5

## Context

M4-T09 closed autosave by showing that storage is write-through, and named the one thing that
genuinely is not covered: state that would live only in the GUI — the last project opened, a
colormap, a detector an operator always uses. That is settings, and it is this task.

The roadmap asked for two scopes, "application scope + project scope", without saying how they
relate.

## Decision

### 1. Two stores, split by what the preference is *about*

- **Application scope** — `$XDG_CONFIG_HOME/nanoscope/settings.json`, i.e.
  `~/.config/nanoscope/settings.json`. Preferences that follow the *operator*: theme, colormap,
  recent projects. This is a Linux desktop application (ADR-0002), so the XDG basedir spec is the
  convention, not one of several, and the file is somewhere a user already knows to look and a
  backup tool already collects.
- **Project scope** — a `settings` table in the project database (schema v4). Preferences about
  *the work*: the detector this project's results were made with, its default labels. They belong
  in the directory, so they travel with a copied or moved project (ADR-0003).

### 2. Reads merge, project first; writes name their scope

A project that states something is stating it about itself, so it wins.

Writing is different: *"save this preference"* without saying where is a question, not an
instruction. Guessing wrong is bad in both directions — one project's choice leaking into every
other project the operator opens, or a global preference hidden inside one directory where the
operator will not find it again. So `Settings.set` takes a `Scope`, defaults to `APPLICATION` (a
preference expressed with no project in mind is about the person expressing it), and **raises** if
`PROJECT` is asked for with no project open rather than silently falling back.

`scope_of(key)` exists so a settings dialog can say *"this project overrides your default"* instead
of showing a value with no explanation.

### 3. Values are JSON, not strings

A store that returns everything as text makes every reader parse it back, and one of them gets it
wrong — `"False"` is truthy. JSON keeps a boolean a boolean, a number a number and a list a list,
in both stores.

`None` stored in a project is an *answer*, not an absence, so the merge asks whether the project
**has** the key rather than what it returns for it.

### 4. Neither store validates what a key means

No schema of known settings, no defaults table. A setting's meaning belongs to whoever reads it,
and a store that knew every key would have to be edited by every feature that adds one — including
features in `gui/`, which this layer must not know about.

### 5. The application file is written by replacement

A temporary file in the same directory and an atomic `os.replace`. A settings file truncated by a
crash mid-write is a preferences reset for somebody who did nothing wrong, and this costs one line.

A file that is already corrupt reads as empty rather than raising: the application must still
start. It is **not** deleted — an operator who hand-edited it can still see what they wrote.

## Consequences

**Positive**

- The two questions a preference can answer — "what do *I* like" and "how was *this work* done" —
  have different homes, and copying a project takes the second with it.
- `SettingsStore` is the second port to pay out `core/ports`'s promise, and it is satisfied by two
  genuinely different implementations, which is what keeps a port honest.
- A settings dialog can explain itself, because the merged view knows where each answer came from.

**Negative**

- Two stores mean a preference can be in the wrong one, and only the operator knows which is wrong.
  Mitigated by `scope_of` making it visible rather than by a rule that guesses.
- The JSON file is read from disk on every access. It is a few hundred bytes, and a cache would
  mean the last process to exit wins when two windows are open.
- Nothing validates keys, so a typo is a silently unread preference. The alternative is a registry
  every feature edits; the honest mitigation is that a caller passes a default it can live with.

**Neutral**

- Schema version 4, the fourth step through ADR-0039's mechanism.

## Alternatives considered

| Alternative | Why not |
|---|---|
| One store for everything | Either project choices follow the operator to other projects, or personal preferences are trapped in one directory |
| Project settings in `project.json` | ADR-0038 keeps the manifest to identity; a manifest rewritten on every preference is a manifest that can be lost mid-write |
| `configparser` / INI | Strings only, so every reader parses — §3 is exactly what INI cannot do |
| A `platformdirs` dependency | A dependency for `~/.config` on a Linux-only application |
| A typed settings schema with declared defaults | Every feature edits one registry, including GUI features this layer must not know about |
| Silently fall back to application scope when no project is open | Puts one project's choice in front of every other project, invisibly |

## Compliance

- `tests/unit/test_settings.py` covers precedence, a `None` that is an answer, the refused
  project-scope write, `scope_of`, type round trips, a corrupt file, and the XDG path.
- `tests/integration/test_project_settings.py` proves a project's settings travel with its
  directory and that the merge works over a real database.
- Writing the application file leaves no temporary behind — asserted, because that is what makes it
  atomic.
- A v3 database gains the table by migration with its rows intact.

## References

- ADR-0046 (write-through storage needs no autosave) — the trigger that named this task
- ADR-0002 (Qt6 desktop, Linux) — why XDG is *the* convention here
- ADR-0003 (projects are directories) — why project settings live in the project
- ADR-0039 (the schema and its migrations) — the mechanism that carried v4
