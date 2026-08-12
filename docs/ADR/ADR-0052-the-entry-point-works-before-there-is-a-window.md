# ADR-0052 — The entry point works before there is a window

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** operator + agent (M5-T01)
- **Affects:** `app/` · M5 · every task that puts a widget on this wiring

## Context

PROJECT_RULES §2.7: *"One composition root. All wiring happens in `app/`. Nothing else constructs
infrastructure objects."* M4 built eight things that need constructing — a repository, a device
manager, a job runner, two settings stores, a log configuration, a model registry, a command stack
— and until now **every caller of them was a test that built them by hand.**

M5's first exit criterion is *"`nanoscope` launches on Linux and opens a project created in M4"*.
The window that would show it is M5-T02, and PySide6 is not yet a dependency of this project at
all.

## Decision

### 1. The entry point works headless, today

`nanoscope --project PATH` opens a project, prints what is in it, and exits. `--devices` lists the
hardware and says which would be chosen. `--gui` returns a sentence saying the window arrives in
M5-T02.

An entry point that only works once a window exists **cannot be run in CI**, and cannot be run by
the operator who most needs it: the one whose project will not open. Opening a project and saying
what is in it needs no display; showing it does.

This also splits the M5 criterion in two, and the half that does not need a window is
green now rather than at the end of the milestone.

### 2. A container, not a framework

`Nanoscope` holds what is expensive or stateful — the application settings, the device manager, the
job runner, the command stack, the open project — and does the two things neither component can do
alone: **resolve the device from the stored preference**, and **open a project as one act** (open
the repository, attach its log, read the integrity report).

Not a DI framework and not a provider registry. The wiring is a handful of lines, and building a
framework to hold a handful of lines is the thing this project keeps deciding not to build
(M2-T08's ports, ADR-0041's use cases, ADR-0046's autosave).

`settings` is a **property**, rebuilt per access, because the project half of the merged view
changes when a project opens or closes and a cached one would answer for a project that is no
longer open.

### 3. Our errors are messages; anything else keeps its traceback

A `NanoscopeError` from the entry point becomes one line on stderr and exit code 1. Everything else
propagates with its stack.

ADR-0030 built exactly this distinction — *"`except NanoscopeError` is the way to catch 'the
library said no' without also catching the bugs"* — and this is the first user-facing surface able
to use it. Printing a traceback at an operator for a directory that is not a project is an
application blaming its user for its own diagnostics.

### 4. The integrity report is *shown*

ADR-0040 decided the check reports and changes nothing, and ended on the obligation that **something
has to read the report**. `open_project` has returned one since M4-T03 and nothing had read it. The
CLI prints it: missing files named with the reminder that their rows are kept, untracked files
named with the reminder that nothing was imported, and a clean project saying so in one line.

### 5. `print` is allowed in exactly one module

`T20` — no `print` in library code — stays selected everywhere else, and M2-T11 deleted thirteen
`print` calls to satisfy it. But **a command-line program's stdout is its user interface**, not a
diagnostic channel: a CLI that logs instead of printing has no output.

The ignore is scoped to `nanoscope/app/main.py`, the one module with a terminal on the other end,
and the logger is used beside it for what belongs in a log.

### 6. Closing a project takes its history and its log with it

`close_project()` detaches the project log (ADR-0051) and **clears the command stack** (ADR-0045):
undo is a session, and a stack whose commands refer to another project's rows is worse than no
history. `close()` shuts the job runner down *before* closing the project, so a job holding the
repository does not find it closed underneath itself.

## Consequences

**Positive**

- One place constructs adapters, so "where does the settings file live" has one answer.
- The application can be run, and a project inspected, without a display — which is what CI, a
  support request and a headless server all need.
- ADR-0040's obligation is discharged rather than inherited again.
- M5-T02 fills a branch rather than inventing a startup path.

**Negative**

- Two entry-point behaviours to keep working — headless and windowed — for the length of M5. The
  headless one is small, tested, and the only one CI can exercise.
- `Nanoscope` will grow as M5 adds view models. The line to hold is the one in §2: it constructs
  and it wires; it does not decide.

**Neutral**

- The console script is declared now, so `nanoscope` is a command as soon as the package is
  installed. `python -m nanoscope` runs the same function, for a checkout with no install.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Ship the entry point with the window (M5-T02) | Nothing runnable in CI, and no way to inspect a project that will not open |
| A DI framework or service locator | A framework to hold eight lines of wiring |
| Construct adapters in the main window | Exactly what PROJECT_RULES §2.7 forbids, and what makes a widget untestable |
| Let exceptions propagate to the terminal | An operator reads a traceback for a typo in a path |
| Log instead of printing, to satisfy `T20` | A command-line program with no output |
| Cache the merged `Settings` | It answers for a project that has since been closed |

## Compliance

- `tests/integration/test_entry_point.py` covers the container's construction, the project log
  attaching and detaching, the undo history cleared with the project, a second project replacing
  the first, the device preference honoured and a nonsense one ignored with a warning — and, on the
  CLI, a project summarised, the integrity report shown, a refusal that is a sentence with exit 1,
  `--devices`, `--gui`, and argparse keeping exit code 2 for usage errors.
- No module outside `gui/` imports Qt (M4-T15's guard), which includes this one.
- `T20` remains selected for every other module in the package.

## References

- PROJECT_RULES §2.7 (one composition root) · §3 (no `print` in library code)
- ADR-0030 (a typed error taxonomy) — the distinction §3 uses
- ADR-0040 (the repository reports) — the obligation §4 discharges
- ADR-0045 (undo is a session) · ADR-0051 (a log must not live inside what it reports on) — §6
- `docs/Roadmap.md` M5 exit criteria
