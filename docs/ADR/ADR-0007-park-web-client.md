# ADR-0007 — The React web client is parked; the desktop app is the product

- **Status:** **Superseded by ADR-0012** (2026-08-04) — the client is deleted, not parked.
  The reasoning below stands and is why it was never finished; only the disposal changed.
- **Date:** 2026-08-03
- **Affects:** `frontend/`, `docs/archive/plan-frontend-react-client.md` · M1, M5

## Context

`frontend/` contains a complete React + TypeScript + Vite + Tailwind client — 13 source
files, upload zone, config panel, result viewer, statistics panel, SVG histograms — built
against `POST /analyze` on a FastAPI backend **that was never written**. There is no
server module in the repository.

The gap is not small. The frontend expects fields Python does not produce
(`masks_preview_b64`, `particle_count`), sends config fields Python does not accept
(`modality`, `nm_per_pixel`), and assumes server-side rendering of previews. Python's
`PipelineResult` carries NumPy masks and a pandas DataFrame, neither JSON-serializable.
Closing it means designing a wire format, a serialization layer, model lifecycle
management in a request context, preview rendering, and CORS/error handling — a
substantial project on its own.

Meanwhile the product direction is now explicit: a Linux desktop application (ADR-0002).
And `frontend/node_modules` is 2 800 tracked files — 98% of the repository (D-19).

## Decision

**Park the web client.**

- The desktop application is the only delivery channel for v1.
- `frontend/` is **not deleted**. It remains in the tree, untouched and unmaintained,
  marked as parked in the documentation.
- `frontend/node_modules` is untracked immediately (M1-T01). The source stays.
- No backend is written. The old `plan.md` is a historical specification, not a plan;
  M1-T01 moved it to `docs/archive/plan-frontend-react-client.md`.
- No further work goes into the client — no fixes, no dependency updates, no CI.
- Deleting the directory outright is a separate decision that needs the operator
  (`STATE.md` B5, backlog B-041).

## Consequences

**Positive**

- The team builds one UI instead of two, and one contract instead of two.
- The serialization boundary — masks, DataFrames, previews — simply disappears for v1;
  the desktop app calls the domain in-process.
- Removes the pressure to design an HTTP API before the domain model is stable.
- Removes 78 MB and 98% of tracked files from the repository.

**Negative**

- Real work is shelved. The React components are decent and represent genuine effort.
- No browser-based access, no remote use, no thin-client scenario. If that is ever
  needed, it starts from a frozen client and a domain that never had to be serializable.
- A parked directory rots: dependencies go stale, and it will confuse future readers
  unless the documentation keeps saying it is parked.

**Neutral**

- The React types (`frontend/src/types/pipeline.ts`) remain a useful reference for what a
  future API contract would need — including the fields Python currently lacks.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Write the FastAPI backend and ship the web client | A large project (wire schemas, serialization, model lifecycle, preview rendering, errors) for a delivery channel that does not match the workload: local GPU, large local files, offline lab machines. |
| Ship both UIs | Two clients, two contracts, one developer. The domain would be shaped by the HTTP boundary rather than by the science. |
| Delete `frontend/` now | Irreversible-feeling to the owner, and unnecessary — untracking `node_modules` captures nearly all the benefit at zero risk. The decision belongs to the operator. |
| Reuse the React client inside a webview | Inherits every downside of the web architecture plus an embedding layer, and gains nothing Qt does not already give. |

## Compliance

- No task in M1–M9 modifies `frontend/`, except M1-T01 untracking `node_modules`.
- CI does not build or test the frontend.
- `docs/Architecture.md` §6 and `docs/Backlog.md` B-041 keep the parked status visible.
- If this decision is reversed, it is reversed by a new ADR that supersedes this one.

## References

- `PROJECT_CONTEXT.md` §4, §11, §15 "frontend/backend contract is incomplete"
- `docs/audit/2026-07-28-baseline-audit.md` D-19
- `docs/archive/plan-frontend-react-client.md` (historical frontend specification)
- ADR-0002
