# CURRENT TASK

**ID:** `M1-T09`
**Title:** Clean notebooks
**Milestone:** M1 — Repository hygiene & quality gates
**Status:** selected — not started
**Branch to use:** `chore/notebooks`
**Estimated size:** S, unless B5 says delete
**Risk to scientific output:** none — no production path may import a notebook
**Selected:** 2026-08-04
**Partially blocked:** **B5** — see below

---

## Why this task is next

Two notebooks account for **8.7 MB** of a repository whose entire tracked source is
2 021 lines: `afm_gold_nanoparticles.ipynb` (6.5 MB) and `preprocessing.ipynb` (2.2 MB).
Almost all of it is embedded output — base64 PNGs of plots, committed and re-committed
every time a cell was run.

They are also the one thing `pre-commit run --all-files` is still red on (M1-T07, M1-T08).
That red is currently correct and expected, which is the worst state for a gate to be in:
a check that is known to fail teaches people to skim past it.

PROJECT_RULES §7 already decides the substance: notebooks are experiments, not interfaces;
they live in `notebooks/`, they are committed without outputs, and no production code path
may depend on them. `nbstripout` is installed and configured (M1-T07) but has never been
run over the existing files.

---

## Scope

**In scope**

1. Strip outputs from both committed notebooks with the already-configured `nbstripout`
2. Move them to `notebooks/` — the directory M1-T01 deleted for being empty, now earning
   its existence
3. Add a short `notebooks/README.md`: what these are, that they are experiments, that
   nothing may import them, and that outputs are stripped on commit
4. Verify no production path imports or references them (`grep`, and check
   `PROJECT_CONTEXT.md`)
5. Measure and record the repository-size delta — tracked size before/after
6. Confirm `pre-commit run --all-files` is **green** afterwards, or state precisely what
   remains and why
7. Check whether the notebooks still execute against current `src/` — **do not fix them if
   they do not.** Record the finding; a broken experiment notebook is M2/M3 information,
   not this task's problem

**Out of scope**

- Deleting the notebooks (**B5** — the operator's call, not the engineer's)
- Rewriting notebook content, updating them to a new API, or making them run
- `frontend/` and `preprocess_batch.py`, the other two parts of B5
- Rewriting git history to reclaim the 8.7 MB already in it (see the note below)

---

## The B5 question this task needs answered

**Stripping outputs is safe and sanctioned by PROJECT_RULES §7 — that part needs no
decision and should proceed regardless.** But stripping is only worth doing if the
notebooks are staying. If B5 says *delete*, this task is one `git rm` and the rest is
wasted work; if it says *archive*, they move to `docs/archive/` instead of `notebooks/`.

Proposed default if no answer arrives: **strip and move to `notebooks/`**, because it is
reversible and loses nothing. Deleting is not reversible from a working tree, and is the
owner's decision.

Worth knowing before deciding: stripping outputs shrinks the *working tree*, not the
repository. The 8.7 MB is already in git history and stays there until someone rewrites it
— which is backlog item B-040, not this task.

---

## Definition of done

- [ ] Both notebooks carry no outputs and no `execution_count`
- [ ] Both live in `notebooks/`, with a `README.md` stating they are experiments
- [ ] Nothing in `src/`, `tests/` or `docs/` references their old paths
- [ ] `pre-commit run --all-files` green, or the remainder named and justified
- [ ] Tracked-size delta measured and recorded
- [ ] `pytest` still green; golden still zero drift (it must be — nothing importable moved)
- [ ] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md`, `docs/Development.md` §8 updated
- [ ] Commit: `M1-T09: strip notebook outputs and move them to notebooks/`

---

## Plan

1. Branch `chore/notebooks`
2. Record the before state: tracked size, per-notebook size, output cell counts
3. `uv run nbstripout afm_gold_nanoparticles.ipynb preprocessing.ipynb`
4. `git mv` both into `notebooks/`; write `notebooks/README.md`
5. `grep -rn` for the old filenames across the repo; fix any reference
6. Run `pre-commit run --all-files` — **from a clean tree**, per the warning in
   `Development.md` §4
7. Run the full gate; update docs; commit; advance `CURRENT_TASK.md` to `M1-T10`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Stripping destroys results the operator still needs.** The outputs are plots of real experiments; some may not be reproducible without the original `data/` files. | Check before stripping whether either notebook contains a figure that is not regenerable from committed code plus local `data/`. If so, export it to `docs/` or `images/` as a file *before* stripping, and say so. This is the one irreversible part of the task. |
| B5 is answered "delete" after the work is done | The work is 15 minutes and the strip is reversible via git. Do not over-invest; do not write tooling around it. |
| A notebook is silently a production dependency | `grep` for imports of the notebook names and for `%run`. PROJECT_RULES §7 forbids it, but the rule postdates the notebooks. |
| The 8.7 MB "reclaimed" is reported as a repository saving | It is not. History is unchanged; `.git` stays 81 MB. Report the working-tree delta only, and point at B-040. |

---

## Notes for the next session

After T09 only **M1-T10** (`make check`) remains, and M1 closes.

Two things are carried, neither of them a task:

- **CI is green** (M1-T08), after four runs that each failed differently. Worth reading
  the entry in `Progress.md` before trusting a "verified locally" claim again.
- **B1, the package name**, is the only thing blocking M2 — open since M0, and every M2
  task depends on it.
