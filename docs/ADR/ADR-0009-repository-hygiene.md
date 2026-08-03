# ADR-0009 — No build artifacts, weights, or datasets in git

- **Status:** Accepted
- **Date:** 2026-08-03
- **Affects:** `.gitignore`, CI, pre-commit · M1

## Context

Measured on `11e0ecc`:

| Metric | Value |
|---|---|
| tracked files | 2 854 |
| of which under `frontend/node_modules` | **2 800 (98.1%), 78.3 MB** |
| `.git` | 81 MB |
| model weights tracked | `yolov8s-world.pt`, **staged in the index** while deleted from the working tree |
| notebooks committed with outputs | 6.5 MB + 2.2 MB |
| `main.ipynb` at the time of the audit | 0 bytes, tracked, not valid JSON |

`.gitignore` covers `data/`, `checkpoints/`, `dataset/` — and not `node_modules/`,
`output/`, `*.pt`, or `.zip`. It *does* ignore `plan.md` and `.claude/`, which prevents
sharing project planning and agent configuration.

The practical consequence is that no diff, log, blame or bisect is readable, and a 26 MB
checkpoint is one `git commit` away from being permanently in history — any commit made
without an explicit pathspec will include it.

## Decision

**Git holds source and documentation. Nothing else.**

Never committed:

- dependency directories — `node_modules/`, `.venv/`, `site-packages`
- model weights — `*.pt`, `*.pth`, `*.onnx`, `*.safetensors`
- raw or derived data — `data/`, `dataset/`, `output/`, `checkpoints/`
- caches and build output — `__pycache__/`, `*.pyc`, `.ruff_cache/`, `.pytest_cache/`,
  `.mypy_cache/`, `build/`, `dist/`, `*.egg-info/`
- archives — `*.zip`
- notebook outputs — notebooks are committed stripped

Always committed:

- source, tests, documentation
- **agent and editor configuration** (`.claude/`) — shared, not personal
- deterministic test fixtures, generated in code (`tests/characterization/phantoms.py`),
  never as binaries

Enforced mechanically:

- `check-added-large-files` in pre-commit, limit 1 MB (M1-T07)
- `nbstripout` in pre-commit (M1-T09)
- CI asserts the tracked-file count stays under a threshold

**History is not rewritten by this decision.** The 78 MB already in history stays until a
separate, operator-approved task (backlog B-040) — rewriting history affects the remote
and every clone.

## Consequences

**Positive**

- Diffs become reviewable; `git log`, `blame` and `bisect` become usable again.
- Clone size stops growing.
- No risk of publishing a proprietary model or the operator's scan data by accident.
- Agent configuration becomes shareable, so tooling behaviour is reproducible across
  machines.

**Negative**

- `git clone` no longer gives a runnable environment: weights must be downloaded and
  `npm install` / `uv sync` run. This must be documented, and it is (`Development.md`).
- Notebook diffs lose their outputs, so a reviewer cannot see results in the diff.
  Accepted — 8.7 MB of committed output is the alternative.
- Existing clones will show `node_modules` as untracked after M1-T01, which looks alarming
  until explained.
- The 78 MB stays in history for now: clone size does not shrink, it only stops growing.

**Neutral**

- Test data must be generated rather than stored, which is already the practice.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Git LFS for weights | Adds infrastructure and a hosting bill for artifacts that are downloadable from upstream and change rarely. Weights belong in `checkpoints/`, fetched by a documented command. |
| Rewrite history now (`filter-repo`/BFG) | The right long-term fix, but it invalidates every existing clone and rewrites the remote. Needs the operator's explicit approval — tracked as B-040. |
| Commit `node_modules` for reproducible frontend builds | The frontend is parked (ADR-0007), and lockfiles already provide reproducibility. |
| Commit a real SPM scan as a fixture | Tempting for parser tests (M3-T16), but it is the operator's data and their call (B6). Phantoms cover the rest. |

## Compliance

- `git ls-files | wc -l` < 100 after M1-T01.
- `git ls-files '*.pt' '*.pth' '*.onnx' | wc -l` == 0.
- Pre-commit rejects any staged file over 1 MB.
- CI fails if the tracked-file count exceeds the threshold.

## References

- `docs/audit/2026-07-28-baseline-audit.md` D-19
- `docs/CURRENT_TASK.md` (M1-T01)
- `docs/Backlog.md` B-040
