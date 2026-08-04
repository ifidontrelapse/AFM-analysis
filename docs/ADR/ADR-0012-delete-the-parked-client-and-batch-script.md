# ADR-0012 — Delete the parked React client and the broken batch script

- **Status:** Accepted
- **Date:** 2026-08-04
- **Supersedes:** ADR-0007 (park the web client)
- **Affects:** `frontend/`, `preprocess_batch.py` · closes `STATE.md` **B5** · M2-T13

## Context

ADR-0007 parked the React client rather than deleting it, explicitly because *"deleting the
directory outright is a separate decision that needs the operator"*. That decision was
tracked as **B5** and has now been made: the operator authorised deletion on 2026-08-04.

Two artefacts are covered.

**`frontend/`** — 21 tracked files: a React + TypeScript + Vite + Tailwind client written
against `POST /analyze` on a FastAPI backend that was never written. Everything ADR-0007
said about it remains true: it expects fields Python does not produce
(`masks_preview_b64`, `particle_count`), sends config fields Python does not accept, and
assumes server-side preview rendering. It has never run against a real server.

**`preprocess_batch.py`** — a CLI batch converter that has failed on **every input file**
since commit `e8caf25` (defect **D-02**): `load_afm` was changed to return `AFMRawData`,
the script still unpacks a 3-tuple, and it reports the resulting failure as
`0 converted, N failed`. It has been broken for the entire period covered by the audit and
nobody noticed, which is the strongest available evidence that nothing depends on it.

## Decision

**Delete both.**

- `frontend/` is removed from the working tree. `docs/archive/plan-frontend-react-client.md`
  stays — it is the historical specification and costs 12 KB.
- `preprocess_batch.py` is removed. Batch processing remains out of scope for v1
  (backlog **B-001**); when it returns it will be an entry point over the application layer,
  not a script that imports `src` directly.
- Both remain in git history. Deletion here means "not in the working tree", not
  "destroyed" — `git show 291b09f:preprocess_batch.py` recovers either at any time.
- The legacy exclusions that existed only for `preprocess_batch.py` are removed from
  `pyproject.toml`, `.pre-commit-config.yaml` and the CI workflow in the same commit.

## Consequences

**Positive**

- The repository contains only code that is either maintained or explicitly scheduled.
  A parked directory rots and confuses readers; ADR-0007 itself listed this as a cost it
  was accepting.
- `preprocess_batch.py` was the only file outside `src/` excluded from the blocking lint
  and format checks. Removing it shrinks the legacy carve-out to exactly one path, `src/`,
  which M2 then dissolves — the exclusion becomes a single, temporary, well-understood
  thing rather than a growing list.
- Eight ruff findings disappear because the file does: 117 → 109, all of them now in `src/`.
- No more explaining, in every document, that a directory is deliberately dead.

**Negative**

- Genuine work is removed from view. The React components were competent, and the type
  definitions in `frontend/src/types/pipeline.ts` were the only written description of what
  a future API contract would need — including the fields Python currently lacks. That
  description now exists only in git history and in
  `docs/archive/plan-frontend-react-client.md`.
- If a browser client is ever wanted, it starts from a specification rather than from
  running code.

**Neutral**

- No production path changes. Nothing in `src/`, `tests/` or the notebooks imports either
  artefact — verified by grep before deletion, not assumed.
- The desktop application (ADR-0002) remains the only delivery channel for v1; that was
  already true under ADR-0007.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Keep parking `frontend/` (status quo, ADR-0007) | The operator has now made the call the earlier ADR deferred. Parking was always a way of not deciding. |
| Move `frontend/` to `docs/archive/` | Archiving source code makes it look retrievable-and-maintained while being neither. Git history already provides retrieval, with better fidelity. |
| Keep `preprocess_batch.py` and fix it (M2-T13's other option) | Fixing a script nobody has successfully run since `e8caf25` would produce a CLI with no stated requirements, competing with the batch feature deliberately deferred as B-001. |
| Delete `docs/archive/plan-frontend-react-client.md` too | It is a specification, not code — the cheapest possible record of what the client was meant to do. 12 KB is not worth the loss. |

## Compliance

- `docs/Architecture.md` §6 no longer lists the web client as parked-but-present.
- `docs/Backlog.md` B-041 is closed as `rejected`, referencing this ADR.
- **M2-T13** narrows to its remaining half: the 10 unreachable functions inside `src/`.
- `README.md` and `project.md` no longer describe a frontend that does not exist.
- Reversing this decision means recovering the files from history under a new ADR.

## References

- ADR-0007 (superseded), ADR-0002 (desktop UI is the product)
- `docs/audit/2026-07-28-baseline-audit.md` D-02, D-19
- `STATE.md` B5 · `docs/Backlog.md` B-001, B-041
- Deletion commit: the working tree before it is `291b09f`
