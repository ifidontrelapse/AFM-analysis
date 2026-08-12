# CURRENT TASK

**ID:** `M5-T01`
**Title:** The composition root, and a `nanoscope` that runs before there is a window
**Milestone:** M5 — GUI shell, first task
**Defect:** — · **ADR:** **ADR-0052**
**Branch:** `feat/m5-gui-shell` (M5 opens its own; M4's is closed)
**Status:** **done 2026-08-12.** The record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## Why this task is first

PROJECT_RULES §2.7: *"One composition root. All wiring happens in `app/`. Nothing else constructs
infrastructure objects."* M4 built eight things that need constructing — a repository, a device
manager, a job runner, two settings stores, a log configuration, a model registry, a command stack
— and **every test so far has constructed them by hand.** That is fine in a test and wrong in an
application: it is how two places end up deciding where the settings file lives.

Every M5 task after this one puts a widget on top of that wiring. Doing it in the reverse order is
how a main window ends up owning a `DeviceManager`.

---

## The decisions this task has to make

**1. Does the entry point need Qt?** **No — and that is what makes it testable.**

PySide6 is not even a dependency of this project yet. A `nanoscope` that only works once a window
exists cannot be run in CI, and cannot be run by an operator debugging a project that will not
open. So the console script works **headless today**: `nanoscope --project PATH` opens a project,
prints what is in it, and exits. Launching a window is a branch the next task fills, and asking for
one without PySide6 installed gets a sentence rather than a traceback.

**2. Is the container a class or a module of functions?** A small class — `Nanoscope` — holding
what is expensive or stateful (the settings store, the device manager, the job runner, the open
project) and nothing else. Not a DI framework, not a registry of providers: **the wiring here is
eight lines, and a framework to hold eight lines is the thing this project keeps deciding not to
build.**

**3. Where does ADR-0040's obligation land?** Here. `open_project` has returned an integrity report
since M4-T03, and the ADR ended by saying *a report nobody reads is a report that did nothing*.
This is the first caller that can read it, so **the CLI prints it** — missing files named, untracked
files named, and a clean project saying so in one line.

**4. What does a failure look like?** A sentence and a non-zero exit — never a traceback. A
`NanoscopeError` is *the library saying no about the operator's data*, and ADR-0030 built that
distinction so a surface like this one could use it: our errors are messages, anything else is a
bug and keeps its traceback.

**5. Does the console script get declared now?** Yes — `[project.scripts]` in `pyproject.toml`. A
script nobody can invoke is a function; the criterion for M5 is that **`nanoscope` launches**, and
this is the half of it that does not need a window.

---

## Scope

**In scope**

1. `app/container.py` — `Nanoscope`: settings, device manager, job runner, logging, open/close
   project, and nothing else
2. `app/main.py` + `app/__main__.py` — argv → actions, readable failures, exit codes
3. `pyproject.toml` — the `nanoscope` console script
4. **ADR-0052** — headless-first entry point, a container rather than a framework, our errors are
   messages
5. Tests: opening a real project, an integrity report printed, a directory that is not a project
   refused with an exit code, `--version`, `--device`, and no Qt anywhere

**Out of scope**

- **The main window** — M5-T02, which fills the `--gui` branch
- **Adding PySide6 as a dependency** — with the window that needs it
- **A settings dialog, a log panel, a project explorer** — M5-T04, T08, T09

---

## Definition of done

- [x] `Nanoscope` constructs everything M4 built, in one place
- [x] `nanoscope --project PATH` opens, prints, and exits cleanly; `--gui` says what is missing
- [x] The integrity report is *shown*, discharging ADR-0040's closing obligation
- [x] ADR-0052
- [x] Tests, headless, including a refusal with a readable message
- [x] `make check` green — 849 tests, golden byte-identical
- [x] Docs, the ADR index, `Roadmap.md`
- [x] Commit: `M5-T01: one place that constructs everything`

---

## What it turned up

**Reading the version at import time cost eleven modules, and M2-T09's guard caught it in the same
run.** `importlib.metadata` pulls in `zipfile` and `email`, taking `import nanoscope.core.entities`
from 250 modules to 261 — past the bound written *precisely* so a new dependency could not sneak in
under the numpy noise floor. The version is now read through a PEP 562 module `__getattr__`: the
cost lands on whoever asks, and the asker is the CLI, which does not care.

**`print` needed one scoped exception, and it needed it twice.** Ruff's `T20` and M2-T11's own test
both forbid `print` in library code, and both are right — a command-line program is the exception,
because its stdout is a user interface rather than a diagnostic channel. Written into
`pyproject.toml` and into the test, each with the reason, and scoped to the single module that has
a terminal on the other end.

**Two of M4's three inherited obligations could not be discharged here**, and both need a widget:
counting annotations before `remove_image` (ADR-0044), and marshalling a job's worker-thread
listener onto the main thread (ADR-0043). They pass to M5-T02 with their ADRs attached.
