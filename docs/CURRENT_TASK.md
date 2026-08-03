# CURRENT TASK

**ID:** `M1-T01`
**Title:** Untrack build artifacts and model weights; rewrite `.gitignore`
**Milestone:** M1 — Repository hygiene & quality gates
**Status:** selected — not started
**Branch to use:** `chore/repo-hygiene` (branched from `main`)
**Estimated size:** S (one session)
**Risk to scientific output:** none — no functional code is touched
**Selected:** 2026-08-03

---

## Why this task is first

Three reasons, in order of weight:

1. **Everything downstream is unreviewable until it is done.** 2 800 of 2 854 tracked
   files — 98.1% — live under `frontend/node_modules`. Any diff, any `git log`, any
   bisect, any code review is drowning in vendored JavaScript.
2. **A model checkpoint is on its way into permanent history.** `yolov8s-world.pt`
   (26 MB, blob `2fa1b38`) is currently *staged in the index* while deleted from the
   working tree — so any `git commit` without an explicit pathspec commits it. Once in,
   it cannot be removed without rewriting history. This is the only task on the roadmap
   with a closing window.
3. **It carries zero scientific risk.** No functional code changes, so no golden number
   can move. It is the correct first move: maximum leverage, zero blast radius.

Reference: `docs/audit/2026-07-28-baseline-audit.md` — defect **D-19** (Critical, hygiene).

---

## Measured starting state

```
tracked files                  2 854
under frontend/node_modules    2 800   (98.1%, 78.3 MB)
tracked *.pt                   1       (yolov8s-world.pt, staged, working tree deleted)
.git                           81 MB
```

`.gitignore` currently covers `data/`, `checkpoints/`, `dataset/`, `.venv`, caches — and
**not** `node_modules/`, `output/`, `*.pt`, `*.zip`. It also ignores `plan.md` and
`.claude/`, which prevents sharing agent configuration (see rule §7).

---

## Scope

**In scope**

1. `git rm -r --cached frontend/node_modules` — untrack, keep on disk
2. Unstage `yolov8s-world.pt` before it enters history
3. Rewrite `.gitignore`:
   - add: `node_modules/`, `output/`, `*.pt`, `*.pth`, `*.onnx`, `*.zip`, `*.egg-info/`,
     `.mypy_cache/`, `build/`, `dist/`
   - keep: `data/`, `dataset/`, `checkpoints/`, `.venv`, `__pycache__`, `*.pyc`,
     `.ruff_cache`, `.pytest_cache`, `.ipynb_checkpoints`
   - **stop ignoring** `.claude/` — agent configuration is shared (PROJECT_RULES §7)
   - decide on `plan.md`: it is a historical frontend spec — un-ignore and move to
     `docs/archive/`, or leave ignored (record the choice in `Progress.md`)
4. Verify no tracked file exceeds 1 MB, except the notebooks already in history
   (those are M1-T09)
5. Commit with the task ID in the subject

**Out of scope** — do not do these here

- Rewriting git history to purge `node_modules` from past commits (needs operator
  approval; the repository has a remote — separate task, separate decision)
- Deleting `frontend/` itself (blocked on decision **B5** in `STATE.md`)
- Stripping notebook outputs (M1-T09)
- Adding pre-commit or CI (M1-T07, M1-T08)
- Touching anything under `src/`

---

## Definition of done

- [ ] `git ls-files | wc -l` < 100
- [ ] `git ls-files frontend/node_modules | wc -l` == 0
- [ ] `git ls-files '*.pt' | wc -l` == 0
- [ ] `frontend/node_modules` still present on disk (untracked, not deleted)
- [ ] `git status` shows no unintended deletions of user files
- [ ] `.claude/settings.json` is tracked
- [ ] Largest tracked file, excluding the pre-existing notebooks, < 1 MB
- [ ] `python tests/characterization/capture.py` still reports zero drift (sanity check —
      it must be unaffected)
- [ ] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [ ] Commit subject: `M1-T01: untrack node_modules and model weights, rewrite .gitignore`

---

## Plan

1. Branch `chore/repo-hygiene` from `main`
2. Record the before-state numbers (`git ls-files | wc -l`, `du -sh .git`)
3. `git rm -r --cached frontend/node_modules`
4. `git rm --cached yolov8s-world.pt`
5. Rewrite `.gitignore`
6. `git add .gitignore .claude/`
7. Verify every checkbox above, including that files still exist on disk
8. Commit
9. Update `STATE.md` (health indicators), `Progress.md`, `TASKS.md`
10. Move `CURRENT_TASK.md` to `M1-T02`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `git rm --cached` mistaken for `git rm` — deletes the user's files | Always `--cached`; verify the directory still exists on disk before committing |
| The uncommitted working-tree changes to `main.ipynb`, `project.md`, `.gitignore` are lost | Do not `git checkout` or `git stash` anything; touch only the listed paths |
| Untracking `node_modules` breaks a teammate's checkout | Nobody depends on vendored `node_modules`; `npm install` regenerates it |
| History still carries the 78 MB | Accepted for now; a history rewrite is a separate, operator-approved decision |

---

## Notes for the next session

After this task, the repository is small enough that `M1-T02` (dev dependencies) and
`M1-T03` (ruff configuration) become trivially reviewable. The natural sequence is
T01 → T02 → T03 → T05 (golden into pytest) → T08 (CI), at which point the quality gate
is real and M2 can begin.

**Do not start M2 before M1-T05 is green.** The characterization harness is the only
thing standing between a large refactor and silent scientific drift.
