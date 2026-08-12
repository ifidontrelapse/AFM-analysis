# ADR-0051 — A log must not live inside the thing whose failure it reports

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M4-T14)
- **Affects:** `infrastructure/logging`, `app/logging` · M5's log panel · supersedes ADR-0013's
  deferred SQLite sink

## Context

M2-T11 gave every module a logger and configured nothing on purpose: *"no library module calls
`basicConfig` or attaches a handler. Configuring logging is the application's decision; a library
that makes it steals it"* (ADR-0013). So every record this project emits currently goes nowhere.

ADR-0013 left one item open by name: *"the SQLite log destination that ADR-0003 and
Architecture §3.1 describe becomes a `logging.Handler` in `infrastructure/logging/`, added when the
database exists."* It exists now. This task had to write that handler or say why not.

## Decision

### 1. No SQLite log sink

The objection that decides it is not performance. It is that **a log must not depend on the thing
whose failure it records.**

The most valuable lines this application will ever write are *"the schema is newer than this
version"*, *"database.sqlite is corrupt"*, *"this directory is not a project"*. A handler that
writes into that project's database has nowhere to put any of them. A log that works only when
everything works is not a log; it is a report of the successful case.

The rest follows and would have been enough on its own: a write transaction per record, contending
for the repository lock the records are *about* (ADR-0043); a table growing with no rotation story
in a schema whose other tables have one; and a support request that begins "send me your database"
rather than "send me the log file".

ADR-0003's list and ADR-0013's deferral are amended on this point.

### 2. Two rotating files, because there are two questions

| Destination | Answers |
|---|---|
| `$XDG_STATE_HOME/nanoscope/nanoscope.log` | *what did the application do* — including everything before a project is open, and every failure to open one |
| `<project>/logs/nanoscope.log` | *what happened to this work* — travels with the directory, which is what ADR-0003 set `logs/` aside for, and is what an operator attaches to a bug report about that project |

State, not config: a log is generated data a user does not edit, so it goes to `~/.local/state`
and not beside M4-T10's settings in `~/.config`.

Opening a second project **detaches** the first's handler. One project's log continuing in
another's file would be worse than no project log at all.

### 3. JSON Lines

One object per record: time, level, logger, message, the exception when there is one, and anything
a caller passed as `extra=` — as *fields*, which is what makes the log structured. `grep` still
works and `jq` works properly, and M5's panel reads `image_id` instead of regexing it back out of a
sentence.

A value that will not serialise becomes its `repr` rather than an error: the record is the thing
being kept, and a log line must never fail because somebody logged an object.

### 4. Only `app/` attaches a handler

`configure_logging()` at startup, before anything can fail, and `attach_project_log(root)` when a
project opens. Library modules keep `getLogger(__name__)` and attach nothing, which is ADR-0013's
rule unchanged.

Handlers are **named**, so configuring twice replaces rather than duplicates — what a restarted
GUI, a reopened project and a test suite all do.

### 5. Rotation is stated, not agonised over

`RotatingFileHandler`, 5 MB, three backups. The number is not the decision; having one is. An
unbounded log on a laptop is a disk-full bug that arrives months later, blamed on something else.

## Consequences

**Positive**

- The failures worth logging are the ones that get logged, including every failure to open the
  place a database-backed log would have lived.
- A project directory carries its own log, so "copy the project and send it" includes the evidence.
- Structured records mean M5's panel filters by level and logger without parsing prose.
- The last open item from ADR-0013 is closed with a decision rather than left as a promise.

**Negative**

- No SQL over log history: "show me every failed import last month" is `jq` over a file rather than
  a query. Acceptable for a single-user desktop application, and the file is greppable by tools the
  operator already has.
- Two destinations mean a record is written twice while a project is open. Two small writes, and
  each answers a different question.
- Rotation can discard old lines. That is what rotation is; the alternative fills a disk.

**Neutral**

- No GUI log panel here. M5 either tails the file or attaches its own handler — both are its
  choice, and neither needs anything from this task beyond the format.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A SQLite handler, as ADR-0013 deferred | The log would be unavailable for exactly the failures worth logging, and would contend with the lock it is reporting on |
| One log file for everything | Either project evidence is missing from a copied directory, or startup failures have nowhere to go |
| Plain-text lines | A GUI panel parses a sentence back into fields it already had |
| `basicConfig` in the library | ADR-0013 settled this: a library that configures logging steals the application's decision |
| No rotation | A disk-full bug months later, blamed on something else |
| A third handler keeping records in memory for the GUI | M5 has no panel yet; it can attach one in four lines when it does |

## Compliance

- `tests/unit/test_log_sinks.py` covers the record shape, lazy `%` arguments arriving rendered, an
  exception captured, `extra=` surviving as fields, an unserialisable value becoming a repr, both
  destinations, project switching and detaching, double configuration, and that rotation is set.
- No module outside `app/` attaches a handler; `tests/unit/test_logging.py` (M2-T11) already
  asserts library modules only emit.

## References

- ADR-0013 (stdlib logging instead of a `LogSink` port) — the deferral this answers
- ADR-0003 (projects are directories) — `logs/`, and the SQLite list amended here
- ADR-0043 (jobs) — the lock a database sink would contend with
- ADR-0047 (settings) — why the application log is *state* and not *config*
