# CURRENT TASK

**ID:** `M4-T14`
**Title:** Logging: two files, and the SQLite sink that ADR-0013 promised
**Milestone:** M4 — Application layer, fourteenth task
**Defect:** — (D-22/D-23 closed in M2-T11; this is the sink half) · **ADR:** **ADR-0051**
**Branch:** `feat/m4-application-layer`
**Status:** **done 2026-08-12.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is next

M2-T11 gave every module a logger and deliberately configured nothing: *"no library module calls
`basicConfig` or attaches a handler. Configuring logging is the application's decision; a library
that makes it steals it"* (ADR-0013). So today every log record this project emits goes nowhere.

ADR-0013 left one thing open, by name: *"the SQLite log destination that ADR-0003 and
Architecture §3.1 describe becomes a `logging.Handler` … added when the database exists."* It
exists. This task either writes that handler or explains why not.

---

## The decisions this task has to make

**1. Does the SQLite sink get written?** **No** — and this is the task's real question, so it gets
argued rather than skipped.

The decisive objection is not performance. It is that **a log must not depend on the thing whose
failure it records.** The most valuable log lines this application will ever emit are "could not
open the database", "the schema is newer than this version", "the project directory is not a
project" — and a handler that writes into that project's database has nowhere to put any of them.
A log that works only when everything works is not a log.

The rest follows: a write transaction per record, contending with the repository lock the log lines
are *about*; a table that grows without a rotation story; and a support request that starts with
"send me your database" instead of "send me the log file".

**2. Then where do records go?** Two rotating files, because there are two questions:

| Where | Answers |
|---|---|
| `$XDG_STATE_HOME/nanoscope/nanoscope.log` | *what did the application do?* — including everything before a project is open, and every failure to open one |
| `<project>/logs/nanoscope.log` | *what happened to this work?* — travels with the directory, which is what ADR-0003 set `logs/` aside for, and is what an operator attaches to a bug report about that project |

**3. Structured how?** **JSON Lines.** One object per record: timestamp, level, logger, message,
and the exception when there is one. `grep` still works, `jq` works properly, and a GUI panel can
parse a line without a regex over a format string. "Structured logs" in the task title is the
requirement; JSONL is the smallest thing that satisfies it.

**4. Who configures it?** `app/` — the composition root, and the only layer allowed to
(PROJECT_RULES §2.7, ADR-0013). Library modules keep their `getLogger(__name__)` and attach
nothing.

**5. What about rotation?** `RotatingFileHandler`, 5 MB × 3. Not a decision worth agonising over,
but worth *stating*: an unbounded log on a laptop is a disk-full bug that arrives months later.

---

## Scope

**In scope**

1. `infrastructure/logging/setup.py` — the JSONL formatter, both rotating handlers
2. `app/logging.py` — `configure_logging(...)` / `attach_project_log(...)`, the composition root's
   half
3. **ADR-0051** — no SQLite sink and why, two files, JSONL, who configures
4. Tests: a record becomes one JSON object per line, an exception is captured, rotation is
   configured, the project handler writes into `logs/`, and configuring twice does not duplicate
   handlers

**Out of scope**

- **A GUI log panel** — M5. It reads the file or attaches its own handler; both are its choice
- **Log levels per module in settings** — M4-T10 stores preferences; wiring one to a logger is a
  line M5 writes when there is a dialog for it

---

## Definition of done

- [x] JSONL records, two rotating destinations, configured only from `app/`
- [x] ADR-0051, including the SQLite sink refused with its reason
- [x] Tests, including the double-configuration case
- [x] `make check` green — golden byte-identical
- [x] `ADR-0013` referenced as answered; docs and the ADR index
- [x] Commit: `M4-T14: a log that only works when everything works is not a log`

---

## What it turned up

**The `_STANDARD` field list wanted to be a hand-written string of attribute names, and it should
not be.** `logging.LogRecord` knows its own attributes — constructing one and reading its
`__dict__` is the same list, derived rather than transcribed, and it does not go stale when the
stdlib adds a field. The three that only appear later (`message`, `asctime`, `taskName`) are added
by name, which is a much shorter thing to keep true.

**mypy went to 7 for one commit.** The registry imports the SAM2 predictor by name, and sam2 ships
neither stubs nor a `py.typed` marker — so `sam2.*` joined the scoped `ignore_missing_imports` list
next to `ultralytics.*`. Scoped per module, as M1-T04 insisted: a blanket ignore would also hide a
typo in a first-party import.
