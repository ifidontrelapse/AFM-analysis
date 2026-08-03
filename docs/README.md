# Documentation index

Start here. Read in this order on a first session.

| Order | Document | What it is |
|---|---|---|
| 1 | [STATE.md](STATE.md) | **Live state** — current milestone, current task, blockers, next steps. Updated every session. |
| 2 | [CURRENT_TASK.md](CURRENT_TASK.md) | The one task in progress, in full detail |
| 3 | [PROJECT_RULES.md](PROJECT_RULES.md) | The project constitution — workflow, architecture rules, code style, quality gate |
| 4 | [Architecture.md](Architecture.md) | Target architecture, strengths/weaknesses of the current code, layer contracts |
| 5 | [Roadmap.md](Roadmap.md) | 10 milestones with goals and exit criteria |
| 6 | [TASKS.md](TASKS.md) | 110 tasks, broken down per milestone |
| 7 | [Development.md](Development.md) | Setup, commands, quality gate, workflow, known traps |
| 8 | [Backlog.md](Backlog.md) | Everything not scheduled yet, with reasons |
| 9 | [Progress.md](Progress.md) | Append-only session log |
| 10 | [ADR/](ADR/README.md) | Architecture Decision Records — 11 records, ADR-0011 awaiting a decision |

## Reference material

| Document | What it is |
|---|---|
| [audit/2026-07-28-baseline-audit.md](audit/2026-07-28-baseline-audit.md) | Phase 0 audit — 24 defects reproduced by execution. Historical, frozen. |
| [audit/characterization-baseline.md](audit/characterization-baseline.md) | The golden-file safety net and how to use it. **Read before touching numerical code.** |
| [`../PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md) | Machine-oriented map of the implementation as it exists today |
| [`../README.md`](../README.md) | User-facing overview — **stale**, rewritten in M9-T01 |

## Conventions

- Documentation is written in English (PROJECT_RULES §3).
- When a document and the code disagree, the code is right and the document gets fixed.
- `STATE.md` and `Progress.md` are updated at the end of every session, finished or not.
