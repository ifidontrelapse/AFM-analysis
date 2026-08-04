# CURRENT TASK

**ID:** `M1-T07`
**Title:** Add pre-commit
**Milestone:** M1 — Repository hygiene & quality gates
**Status:** selected — not started
**Branch to use:** `chore/pre-commit`
**Estimated size:** S–M
**Risk to scientific output:** none — hooks do not edit `src/` unless a hook is configured
to, and this task configures none that do
**Selected:** 2026-08-04

---

## Why this task is next

The gate now works and reports the truth: `pytest` is green (M1-T05, M1-T06), ruff and
mypy are configured and honest (M1-T03, M1-T04). Everything it says arrives **after** the
commit, though, and M1-T01 exists precisely because that is too late — a 26 MB checkpoint
and 2 800 `node_modules` files were already in the index by the time anyone looked.

Pre-commit is the first mechanism in this project that can refuse. It is also the cheap
half of the gate: hooks run in under a second on a normal diff, while CI (M1-T08) takes
minutes and belongs on the server.

M2 is the reason for the timing. It moves twelve modules; it will produce many commits in
a row, and the failure it must not allow is a stray `.pt`, a notebook with 6 MB of
outputs, or an unformatted file slipping through while attention is on the refactor.

---

## Scope

**In scope**

1. `pre-commit` as a dev dependency in `[dependency-groups] dev`, pinned like the others
2. `.pre-commit-config.yaml` with, at minimum:
   - `ruff check --fix` and `ruff format` — same version as `pyproject.toml`, no drift
   - `check-added-large-files` at **1 MB**
   - `end-of-file-fixer`, `trailing-whitespace`, `check-merge-conflict`, `check-yaml`,
     `check-toml`
   - `nbstripout` — notebooks are committed without outputs (PROJECT_RULES §7)
3. Decide, and record, **which checks are not hooks.** Recommended: `pytest` and mypy stay
   out of the commit path. The golden alone is 200 s; a hook that slow gets bypassed with
   `--no-verify`, and a gate people route around is worse than no gate. They belong in CI.
4. Verify each hook actually fires: stage a 2 MB file, an unformatted file, a notebook with
   outputs, a file with no trailing newline — each must be refused or fixed
5. `pre-commit run --all-files` once, deliberately, and **record what it changes**. It will
   touch `src/` — 109 ruff findings are open there, many auto-fixable. See the risk below;
   this task does **not** apply them.
6. Document installation (`pre-commit install`) in `docs/Development.md`, including the
   fact that a fresh clone has no hooks until that command is run

**Out of scope**

- Fixing the 109 ruff findings in `src/` (M2 — and each has to be read, not auto-applied)
- CI (M1-T08)
- Running the golden or mypy as hooks (see item 3 — record the decision, do not implement)
- Notebook relocation to `notebooks/` (M1-T09); this task only strips outputs on commit

---

## Definition of done

- [ ] `pre-commit install` works from a clean clone; hooks fire on `git commit`
- [ ] Every configured hook demonstrated failing on a deliberately bad staged file
- [ ] A commit of an unchanged tree is clean and takes **under ~2 s**
- [ ] `src/` is not reformatted by this task — hooks are added, findings stay open
- [ ] Hook tool versions match `pyproject.toml`; a drift between them is a bug
- [ ] `pytest` still green, golden still zero-drift
- [ ] `docs/Development.md`, `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [ ] Commit: `M1-T07: add pre-commit hooks`

---

## Plan

1. Branch `chore/pre-commit`
2. Add the dependency; write `.pre-commit-config.yaml`
3. `pre-commit install`; commit a deliberately bad file four ways and confirm four refusals
4. `pre-commit run --all-files`, capture the output, then **revert everything it changed in
   `src/`** and record the count in `Progress.md` as the M2 work list
5. Decide the ruff hook's posture — `--fix` or check-only — and write the reason down
6. Update the docs; commit; advance `CURRENT_TASK.md` to `M1-T08`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **`ruff --fix` as a hook silently rewrites the scientific core.** This is exactly what M1-T03 removed `fix = true` for: `ruff check` must answer a question, not edit `src/`. | The hook stages only what is already being committed, so it cannot touch untouched files. Verify this claim rather than assuming it — stage one file in `src/`, commit, and confirm no sibling module changed. If it cannot be constrained, use check-only and let the author run `--fix` deliberately. |
| A slow hook trains everyone to use `--no-verify` | Nothing over ~2 s on the commit path. The golden and mypy go to CI. |
| `nbstripout` mangles a notebook someone cares about | The two committed notebooks (6.5 MB, 2.2 MB) are experiments (PROJECT_RULES §7) and B5 is open on their fate. Strip outputs only; do not reorganise. If B5 says delete, M1-T09 does it. |
| Hook versions drift from `pyproject.toml`, so local and CI disagree | Pin the ruff rev to the installed version and state in the config that the two move together. M1-T08 should read the same version. |

---

## Notes for the next session

After T07: T08 (CI), T09 (notebooks), T10 (`make check`) — then M1 is closed.

**B1 is the only thing blocking M2** and has been open since M0. Every M2 task depends on
the package name, and M2 is the next milestone. It should be answered before M1 closes.

New defect **M3-T20** was filed during M1-T06 (`load_afm(fmt="npy")` fabricates a physical
scale); it is pinned by a test assertion and needs no action before M3.
