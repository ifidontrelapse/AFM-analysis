# CURRENT TASK

**ID:** `M2-T01`
**Title:** Create the package skeleton
**Milestone:** M2 — Domain extraction (behaviour-preserving) — **first task**
**Status:** **done 2026-08-04**. Rewritten for `M2-T02` at the start of the next session.
**Branch to use:** `feat/nanoscope-skeleton`
**Estimated size:** S
**Risk to scientific output:** none — nothing under `src/` is touched, no code moves yet
**Selected:** 2026-08-04

---

## Why this task is next

M1 is closed: the gate is one command, CI runs the same targets, and the golden runs
inside `pytest`. Every remaining M2 task moves scientific code and must prove it moved no
number — that proof now exists mechanically.

Fifteen of the sixteen M2 tasks import from a package that does not exist yet. This
creates it. B1 was answered on 2026-08-04, so the name is settled: **`nanoscope`**
(ADR-0011, Accepted).

---

## Scope

**In scope**

1. `nanoscope/` at the repository root, with the six layer packages from ADR-0011 §Decision
   and `docs/Architecture.md` §3.1: `app`, `core`, `application`, `infrastructure`, `gui`,
   `resources`
2. `py.typed` at the package root — the package is typed, and mypy is already configured
   strict for `nanoscope.*` (M1-T04 wrote that override before the package existed)
3. Distribution renamed `afm-analysis` → `nanoscope`, and `uv.lock` regenerated to match —
   CI runs `uv sync --locked`, which fails on a stale lock
4. `[tool.mypy] files` and `[tool.ruff.lint.isort] known-first-party` extended to the new
   package, so it is checked and its imports sort correctly from the first line
5. `import nanoscope` works, and `make check` stays green

**Out of scope**

- **Moving any code.** `types.py` is M2-T02, `preprocess.py` is M2-T03, and so on. This
  task must not change a single number
- **The editable install and the `pythonpath` hack** — M2-T14. Until then `pythonpath`
  already contains `"."`, so `import nanoscope` resolves without further work
- **The console script and `[build-system]`** — nothing is installed or built yet, and the
  entry point has nothing to launch until M5
- **Sub-packages below the layer level** (`core/entities/`, `core/ports/`,
  `core/science/…`). Each arrives with the code that fills it, in M2-T02…T08. An empty
  directory tree tests nothing and documents a plan `Architecture.md` §3.1 already holds

---

## Definition of done

- [x] `python -c "import nanoscope"` succeeds from the repository root
- [x] The six layer packages exist and import
- [x] `nanoscope/py.typed` exists
- [x] `[project] name = "nanoscope"`, and `uv lock --check` passes — the lock is not stale
- [x] mypy checks `nanoscope` and its strict override is no longer reported as unused
- [x] `make check` green; golden zero drift
- [ ] CI green
- [x] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [x] Commit: `M2-T01: create the nanoscope package skeleton`

---

## Plan

1. Branch `feat/nanoscope-skeleton`
2. Create the package and the six layer `__init__.py` files; one line each stating the
   layer's dependency rule, because that rule is what the directory is *for*
3. Rename the distribution; `uv lock`; **read the lock diff** — anything beyond the project
   name means a re-resolution, and the golden is sensitive to numpy/scipy versions
4. Point mypy and ruff's isort at the package
5. `make check`; push; confirm CI green

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **`uv lock` silently re-resolves numpy/scipy** and the golden moves for a reason that has nothing to do with this task | Read the lock diff before committing. If any version changed, stop — that is a separate, declared decision (PROJECT_RULES §4.1), not a side effect of a rename. |
| Empty scaffolding rots into a plan nobody follows | Only the layers ADR-0011 names. Deeper directories arrive with their code. |
| The rename breaks `uv sync --locked` in CI, which is the whole quality gate | `uv lock --check` locally before pushing; CI proves it after. |

---

## Notes for the next session

`M2-T02` — extract entities and value objects from `types.py`. It is the first task that
moves scientific code, so it is the first real test of the golden as a mechanical gate.

Carried, still not tasks:

- **B-058**: the golden compares CPython exception text. Needs an ADR before any Python
  upgrade.
- **B-054**: the two README figures over 1 MB, the one M1 exit criterion left open.
  Belongs to M9-T01.
