# CURRENT TASK

**ID:** `M4-T11`
**Title:** CSV export — the file somebody opens three months later
**Milestone:** M4 — Application layer, eleventh task
**Defect:** — · **ADR:** **ADR-0048**
**Branch:** `feat/m4-application-layer` — M4 changes no scientific output (PROJECT_RULES §7)
**Status:** **done 2026-08-12**, together with `M4-T09` (ADR-0046) and `M4-T10` (ADR-0047), which
the operator asked for in one session. Each has its own commit and its own ADR; this file carries
the last of the three, and the full record is in `docs/Progress.md` and `docs/TASKS.md`.

---

## What the three tasks decided

**`M4-T09` — there is no dirty state to save (ADR-0046).** Autosave was scheduled before there was
storage to autosave. Every mutating repository method commits before it returns, so a service would
be a timer that flushes nothing — worse than useless, because it would create the impression of
protection where the protection actually lives. What ships is the proof: `test_durability.py`
abandons repositories without `close()` and kills a process with `SIGKILL` between writes. Two
triggers are named that would reverse it.

**`M4-T10` — a preference belongs either to the operator or to the work (ADR-0047).** Two stores,
split by what the preference is *about*. Reads merge with the project first; writes name their
scope, because "save this" without saying where is a question and guessing is wrong in both
directions. Values are JSON so a boolean survives, and a stored `None` is an answer rather than an
absence.

**`M4-T11` — an export is not a copy of the stored table (ADR-0048).** ADR-0042 predicted this
would be nearly free; the format was, and the export was not. Provenance columns in front, more
than one run in one file, a timestamped name — and a refusal to write a CSV that would misrepresent
what happened.

---

## Definition of done

- [x] ADR-0046, ADR-0047, ADR-0048
- [x] `test_durability.py`; settings in two scopes with schema v4; `export_measurements`
- [x] `make check` green — 704 tests, golden byte-identical
- [x] `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, `Architecture.md`, ADR index
- [x] Three commits, one per task

---

## What it turned up

**The `SIGKILL` durability test failed once, and the cause was the working tree changing under it.**
It is the only test here that spawns a process, so it reads the code from *disk* while every other
test reads it from the parent's memory: the parent held schema v3 while the subprocess imported v4
and migrated the fixture forward. Not a property of the application — but worth a docstring, since
the next person to hit it would debug it as one.

**Autosave was the second scheduled task this milestone that turned out not to need building** —
after M2-T08's six ports and ADR-0041's three use cases. The pattern is consistent enough to name:
a task written before the layer beneath it existed is a *hypothesis*, and checking it is part of
doing it.

---

## Notes

The golden held for the eleventh time. **M4-T12** takes the `DeviceManager` — the first task in this
milestone that has to touch torch, and one of the milestone's exit criteria.
