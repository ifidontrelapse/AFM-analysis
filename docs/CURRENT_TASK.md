# CURRENT TASK

**ID:** `M1-T03`
**Title:** Repair the ruff configuration
**Milestone:** M1 — Repository hygiene & quality gates
**Status:** selected — not started
**Branch to use:** `chore/ruff-config`
**Estimated size:** S
**Risk to scientific output:** none, **if** `fix = true` is removed first — see the risk table
**Selected:** 2026-08-03

---

## Why this task is next

M1-T02 installed ruff and ran it for the first time in the project's history. The
configuration is not merely stale — one setting makes the tool actively dangerous:

```toml
fix = true          # `ruff check .` REWRITES source files
show-fixes = true
```

Asking ruff a question currently changes 66 files' worth of code as a side effect. Any
contributor — or agent — who runs the documented `ruff check .` performs an unreviewed
refactor of the scientific core. That must be off before anyone is told to run the gate.

Three further defects, all confirmed by running the tool:

```
warning: The top-level linter settings are deprecated in favour of their counterparts
in the `lint` section:  'ignore' -> 'lint.ignore',  'select' -> 'lint.select'
```

- `target-version = "py311"` on a project that requires `>=3.12` — `UP` rules are
  therefore not proposing 3.12-era idioms
- `known-first-party = ["your_package_name"]` — an unedited template value, so `I001`
  (11 findings in `src/`) is sorting imports against a package that does not exist

Reference: `docs/audit/2026-07-28-baseline-audit.md` — defect **D-20**.

---

## Scope

**In scope**

1. **Remove `fix = true`** (and `show-fixes`, which is meaningless without it). Fixing
   becomes explicit: `ruff check --fix`.
2. Move `select` / `ignore` under `[tool.ruff.lint]`; the deprecation warning must disappear.
3. `target-version = "py312"`.
4. `known-first-party = ["src"]` for now — it becomes the real package name in M2-T01,
   which is why this is a one-line change and not a decision.
5. Review `per-file-ignores`: `"tests/*" = ["T20", "S101"]` references `S101`, but the
   `S` (bandit) rules are not selected. Either select `S` or drop the dead entry.
6. Decide the notebook policy: ruff currently lints `*.ipynb`, which contributes 88 of the
   196 findings. Notebooks are experiments, not interfaces (PROJECT_RULES §7) — exclude
   them from lint, and record that decision here.
7. Re-measure and confirm the `src/` count is unchanged by configuration alone (it should
   stay at 108 — a configuration repair must not silently change what is being measured).

**Out of scope**

- Fixing any finding. All 108 `src/` findings are M2 work, done under the protection of
  the golden file.
- mypy configuration (M1-T04).
- Enabling new rule families — the current selection is fine; changing it would move the
  baseline and make the M2 burn-down unmeasurable.

---

## Definition of done

- [ ] `ruff check .` emits **no** deprecation warning
- [ ] `ruff check .` leaves the working tree unmodified — verify with `git diff --exit-code`
      immediately after running it
- [ ] `ruff check src/ --no-fix --statistics` still reports **108** findings
      (configuration repair, not a rule change)
- [ ] `ruff format --check .` runs without error (it may report files needing formatting;
      that is expected and is not fixed here)
- [ ] Notebook lint policy applied and stated in `Progress.md`
- [ ] `python tests/characterization/capture.py` reports zero drift
- [ ] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [ ] Commit: `M1-T03: repair the ruff configuration`

---

## Plan

1. Branch `chore/ruff-config`
2. Edit the `[tool.ruff]` block: drop `fix`/`show-fixes`, move `select`/`ignore` to
   `[tool.ruff.lint]`, bump `target-version`, fix `known-first-party`, resolve the
   `S101` entry, add the notebook exclusion
3. `ruff check .` → confirm no warning, then `git diff --exit-code` → confirm no rewrite
4. `ruff check src/ --no-fix --statistics` → confirm 108
5. `capture.py` → confirm zero drift
6. Update docs, commit, advance `CURRENT_TASK.md` to `M1-T04`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Running `ruff check .` before removing `fix = true` silently refactors `src/`** | Remove the setting in the *first* edit, before running the tool. If it is run by accident, `git checkout -- src/` restores the tree — the working tree is clean at the start of this task, so nothing else would be lost. |
| Excluding notebooks hides a real problem | The notebook findings are import order and `print` in experiments. They are excluded from *lint*, not deleted; M1-T09 handles notebooks properly. |
| Tightening rules while "repairing" configuration | Explicitly out of scope. The 108 count is the invariant that proves nothing was tightened. |

---

## Notes for the next session

After T03 → T04 (mypy config) → T05 (golden into pytest) → T06 (real I/O test) →
T07/T08 (pre-commit, CI). At that point `make check` is meaningful and M2 can begin.

**Do not start M2 before M1-T05 is green.** The characterization harness is the only
thing standing between a large refactor and silent scientific drift.
