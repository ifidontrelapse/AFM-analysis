# CURRENT TASK

**ID:** `M1-T10`
**Title:** Add a one-command gate
**Milestone:** M1 — Repository hygiene & quality gates — **last task**
**Status:** **done 2026-08-04** — and with it, M1. The next session rewrites this file for
`M2-T01`; the record of what was done lives in `docs/Progress.md`.
**Branch to use:** `chore/make-check`
**Estimated size:** S
**Risk to scientific output:** none — a wrapper around commands that already exist
**Selected:** 2026-08-04

---

## Why this task is next

Everything the gate needs now exists and passes: hooks refuse bad commits (M1-T07), CI runs
the slow half on push (M1-T08), `pytest` is green including the golden (M1-T05, M1-T06), and
`pre-commit run --all-files` is green across the repository (M1-T09).

What is missing is one place that says what "the gate" *is*. Today it is four commands in
`docs/Development.md` §4, a hook list in `.pre-commit-config.yaml`, and a workflow in
`.github/workflows/ci.yml`. Three descriptions of one thing, and they can drift — the M1-T08
near-miss was exactly that (an exclusion declared in two files, nearly three).

This closes M1.

---

## Scope

**In scope**

1. A `Makefile` at the repository root with, at minimum:
   - `check` — the full gate, in the order CI runs it
   - `lint`, `format`, `types`, `test`, `golden` — the pieces, individually runnable
   - `fast` — `pytest -m "not slow"`, the inner loop
   - `help` — the default target; a bare `make` should list what exists, not run 200 s
2. **CI calls the Makefile targets**, so that the workflow and the local gate cannot
   describe different things. This is the point of the task, not a nicety
3. Each target echoes the command it runs, so `make check` teaches the underlying commands
   rather than hiding them
4. `make check` must fail on the first failing step, with a non-zero exit code — verified
   by breaking something on purpose
5. Document it in `docs/Development.md` §4, replacing the four-command block that currently
   stands in for it, and in `PROJECT_RULES.md` §6 if the wording there needs it

**Out of scope**

- `just` instead of `make` — decide in one line, do not survey. `make` is present on every
  Linux machine, which is the stated target platform
- Adding new checks. This task wraps what exists; a target for coverage or `pre-commit` is a
  separate decision
- Fixing the `src/` findings the gate reports

---

## Definition of done

- [x] `make` with no arguments prints the target list and runs nothing slow
- [x] `make check` runs the full gate and passes on a clean tree
- [x] `make check` **fails, with a non-zero exit, on the first broken step** — proven by
      breaking one thing, observing red, reverting
- [x] Every individual target works alone
- [x] `.github/workflows/ci.yml` invokes the Makefile targets; a green CI run proves it
      — **run 14 green, 216 s**, first try, and the CPU-only assertion still passes
- [x] No command is described in two places any more — or, where it must be, the duplication
      is named and justified in a comment
- [x] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md`, `docs/Development.md` updated
- [x] Commit: `M1-T10: add a one-command gate`

---

## Plan

1. Branch `chore/make-check`
2. Write the `Makefile`; keep it under ~40 lines — if it needs more, the gate is too clever
3. Run every target individually, then `make check` whole
4. Break one step deliberately; confirm a red, non-zero result; revert
5. Point CI at the targets; push; confirm the run is green **and still installs the CPU-only
   environment** — the assertion step must survive the refactor
6. Update the docs; commit; push
7. **Close M1**: write the milestone summary into `docs/Progress.md` against
   `docs/Roadmap.md`'s exit criteria, then select **M2-T01** — unblocked, the package is
   `nanoscope`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **The Makefile becomes a second, subtly different gate** — the exact failure this task exists to prevent | CI must call the targets, not re-list the commands. If a target cannot be used in CI, that is a finding to record, not something to work around silently. |
| Make's tab-vs-spaces and shell-per-line semantics produce a target that silently does not fail | Every multi-command target uses `set -e` or `&&`. The deliberate-breakage check in the DoD is what proves it; a gate that cannot fail is the recurring theme of M1-T05, T06 and T08. |
| `make check` runs the 200 s golden and people stop using it | `make fast` exists for the inner loop, documented beside it, exactly as `pytest -m "not slow"` is today. |
| Overbuilding — parameterised targets, `.PHONY` hygiene theatre, colour output | ~40 lines. It is a list of commands with names. |

---

## Notes for the next session

**This is the last task in M1.** After it, write the milestone summary and check it against
the exit criteria in `docs/Roadmap.md`.

**M2 is unblocked.** B1 was answered on 2026-08-04 — the package is `nanoscope`, ADR-0011
is Accepted, and M2-T01 can start the moment M1 closes.

Also carried, neither of them a task:

- **B-058**: the golden is pinned to CPython's minor version, not just to the numerical
  libraries. Needs an ADR before anyone upgrades Python.
- **ADR-0012** deleted `frontend/` and `preprocess_batch.py`. The blocking lint and format
  checks now carve out exactly one path, `src/` — so when M2 dissolves it, the exclusion
  disappears entirely rather than shrinking.
