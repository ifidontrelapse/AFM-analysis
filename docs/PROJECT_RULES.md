# PROJECT_RULES.md — the project constitution

**Status:** active · **Version:** 1.0 · **Created:** 2026-08-03

These rules bind every contributor, human or agent. Code that violates them does not
get merged. When a rule turns out to be wrong, change the rule in a PR — do not
silently ignore it.

---

## 0. Document map

| Document | Purpose | Update cadence |
|---|---|---|
| `docs/PROJECT_RULES.md` | This file. The constitution. | Rarely, by PR |
| `docs/STATE.md` | Live state: milestone, task, blockers, next | **Every session** |
| `docs/CURRENT_TASK.md` | The single task in progress, in full detail | Every task switch |
| `docs/TASKS.md` | Full task breakdown per milestone | When tasks are added/closed |
| `docs/Roadmap.md` | Milestones, goals, exit criteria | Per milestone change |
| `docs/Backlog.md` | Everything not scheduled yet | Whenever an idea appears |
| `docs/Progress.md` | Append-only session log | **Every session** |
| `docs/Architecture.md` | Target architecture and layer contracts | On architectural change |
| `docs/Development.md` | Environment, commands, workflow | On tooling change |
| `docs/ADR/` | One file per architectural decision | On every decision |
| `docs/audit/` | Phase 0 audit + characterization baseline (historical, do not edit) | Frozen |
| `PROJECT_CONTEXT.md` | Machine-oriented map of the **current** implementation | When `src/` changes |

`README.md` is user-facing and currently stale (see `docs/audit/2026-07-28-baseline-audit.md`, D-24).
It is rewritten in milestone **M9**, not before.

---

## 1. Engineering workflow

Every unit of work follows this cycle. No step may be skipped.

```
1 Read       — the task, the affected code, the relevant ADRs
2 Analyze    — what breaks, what depends on it, what the blast radius is
3 Plan       — write the plan into CURRENT_TASK.md before touching code
4 Document   — update Architecture.md / write an ADR if a decision is made
5 Implement  — smallest change that satisfies the task
6 Test       — new tests + full gate green (see §6)
7 Document   — update STATE.md, Progress.md, TASKS.md, PROJECT_CONTEXT.md
```

**Never start implementing immediately.** A session that produces code without a
`CURRENT_TASK.md` entry is a rule violation.

### Session protocol

- **Start of session:** read `docs/STATE.md`, then `docs/CURRENT_TASK.md`.
- **End of session:** update `docs/STATE.md` and append to `docs/Progress.md` —
  even if the task is unfinished, especially if it is unfinished.
- One task in `CURRENT_TASK.md` at a time. Finish or explicitly park it.

---

## 2. Architecture rules

The target layering is defined in `docs/Architecture.md`. The rules that make it real:

1. **Dependency direction is one-way:**
   `gui → application → core` and `infrastructure → core`.
   `core` imports nothing from `application`, `gui`, or `infrastructure`. Ever.
2. **`core` is pure.** No Qt, no torch, no ultralytics, no SQLite, no filesystem
   access, no network, no `print`. NumPy / SciPy / scikit-image / pandas only.
3. **The GUI contains no business logic.** A widget may format, lay out, and emit
   signals. It may not decide *what* to compute, *which* device to use, or *how* to
   measure. If a `gui/` file imports a science module, that is a bug.
4. **Interfaces live in `core/ports/`.** Implementations live in `infrastructure/`.
   Consumers depend on the port, never on the implementation.
5. **No model-specific logic outside its provider.** The strings `"yolo"`, `"sam2"`,
   `"log"` must not appear in `gui/` or in `core/science/`. Selection happens in the
   registry; behaviour lives in the provider.
6. **No device decisions outside `DeviceManager`.** No module calls `torch.cuda.is_available()`
   or hardcodes `"cuda"` / `"cpu"`.
7. **One composition root.** All wiring happens in `app/`. Nothing else constructs
   infrastructure objects.
8. **No global mutable state.** No module-level singletons, no import-time side
   effects, no configuration read from module scope.
9. **Prefer composition over inheritance.** Abstract base classes exist to declare
   ports, not to share implementation.
10. **No God objects.** A class with more than one reason to change gets split.

### Import hygiene

- Package `__init__.py` files re-export names only. They must not import heavy
  subpackages — importing the entity module must not pull in torch, matplotlib, or Qt.
  (This is defect **D-18**: five import cycles caused by exactly this.)
- Heavy optional dependencies (torch, ultralytics, sam2, PySide6) are imported inside
  the function or module that needs them, never at the top of a shared module.
- An automated import-graph test enforces §2.1 and §2.2. It is part of the gate.

---

## 3. Code style

- **Python 3.12+.** Type annotations on every public function, method, and dataclass
  field. `from __future__ import annotations` at the top of every module.
- **English only** — identifiers, comments, docstrings, log messages, exception text,
  commit messages, documentation. (Defect **D-22**: 197 lines of Russian currently
  reach runtime output.) User-facing strings go through the translation catalog, not
  into source strings.
- **Docstrings on every public API**, in the style already used in `src/`: summary,
  `Args:`, `Returns:`, `Raises:`. Document units and coordinate conventions explicitly.
- **No `print` in library code.** Use the structured logger. Ruff rule `T20` enforces this.
- **Explicit errors.** Raise typed project exceptions that name the offending parameter
  and its value. Never let a NumPy/SciPy internal error escape as the public contract.
- Formatting and linting are not opinions: `ruff format` and `ruff check` decide.
  Line length 100.

### Naming and units — non-negotiable invariants

| Convention | Rule |
|---|---|
| Arrays | Indexed `[y, x]`. Always. |
| Detections | Expose `x_px`, `y_px`. Boxes are `(x1, y1, x2, y2)`. |
| Suffixes | `_px` = pixels, `_nm` = nanometres, `_nm2` = square nanometres, `_v` = volts |
| Unknown scale | Physical values are `None`. Never `0`, never the pixel value, never a crash. |
| Heights | Nanometres, after calibration, always. |
| Magic constants | `radius_px = sigma * sqrt(2)` and friends live in one named constant, not in six files. |

---

## 4. Scientific-core rules

The existing pipeline in `src/` is the **Domain layer**. It is the reason this project
exists and it is treated as such.

1. **Do not rewrite the science to make the architecture prettier.** Move it, wrap it,
   type it, test it — but the numerics change only for a stated scientific reason.
2. **Restructuring must not move a single number.** Before and after any refactor:
   ```bash
   python tests/characterization/capture.py    # must report no drift
   ```
3. **A change that moves a golden number requires all four, in the same commit:**
   an ADR, a test that proves the new behaviour, the regenerated golden file, and a
   `Progress.md` entry quantifying the delta.
4. **Never bundle a numerical fix with a refactor.** One commit, one intent.
   Defects **D-03, D-04, D-10, D-12** each get their own commit and their own ADR
   (see `docs/audit/2026-07-28-baseline-audit.md` §5).
5. **A defect that changes scientific output needs operator sign-off** before it is
   fixed. The operator decides what the correct physics is; the engineer decides how
   to implement it.
6. Degenerate inputs (empty, constant, negative, NaN, Inf, 1-D, 3-D) are part of the
   contract. Every numerical entry point states what it does with them, and a test proves it.

---

## 5. Data, storage, and persistence

- A project is a **plain directory**. A user must be able to open it with a file
  manager, copy it, and put it in version control.
- SQLite stores **metadata only**: projects, image metadata, annotations, measurements,
  training history, logs, settings. **No image binaries in the database.** No mask
  bitmaps in the database — masks are files, the database stores paths.
- Every schema has a version and a forward migration. No destructive migrations.
- The project directory layout is fixed:
  `images/ annotations/ results/ exports/ models/ logs/ cache/ database.sqlite`
- Anything under `cache/` must be safely deletable at any time without data loss.
- Export format is **CSV** only, for now.

---

## 6. Quality gate

The following must pass before any merge:

```bash
ruff check .                                   # lint
ruff format --check .                          # format
mypy nanoscope                                 # types
pytest                                         # unit + integration + GUI smoke
python tests/characterization/capture.py       # golden numerical drift
```

- **New behaviour without a test is not done.** Bug fixes start with a failing test.
- Tests are deterministic, CPU-only, and network-free. Seeded RNG or no RNG.
- Model inference (YOLO, SAM2) is not part of the gate — it is not reproducible enough.
  Those paths are covered by contract tests around the provider boundary and by manual
  evaluation runs.
- GUI tests run headless (`QT_QPA_PLATFORM=offscreen`).

---

## 7. Repository hygiene

- **Never commit:** `node_modules/`, model weights (`*.pt`, `*.pth`, `*.onnx`),
  raw scan data, `dataset/`, `output/`, virtualenvs, caches, notebook outputs.
- Notebooks are **experiments, not interfaces**. They live in `notebooks/`, they are
  committed without outputs, and no production code path may depend on them.
- Binary test fixtures do not enter git. Test data is generated deterministically
  (see `tests/characterization/phantoms.py`).
- Agent/editor configuration (`.claude/`) **is** shared and tracked.

### Branches and commits

- Branch naming: `type/short-slug` — `feat/`, `fix/`, `refactor/`, `docs/`, `chore/`, `sci/`.
  `sci/` marks a branch that intentionally changes scientific output.
- One task per branch. One intent per commit.
- Commit subject: imperative, English, ≤72 chars, prefixed with the task ID:
  `M1-T01: untrack node_modules and rewrite .gitignore`
- A commit that changes numerics states the delta in its body.

---

## 8. Documentation rules

- **Every architectural decision gets an ADR.** If you had to choose between two ways
  of doing something and the choice will be expensive to reverse, write the ADR.
- ADRs are immutable once accepted. To change a decision, write a new ADR and mark the
  old one `Superseded by ADR-XXXX`.
- ADR file names: `docs/ADR/ADR-XXXX-kebab-title.md`, using `docs/ADR/TEMPLATE.md`.
- Documentation that contradicts the code is worse than no documentation. When they
  disagree, the code wins and the document gets fixed in the same PR.
- Do not claim a feature exists because a document mentions it. Verify in source.

---

## 9. Changing these rules

Amendments are PRs that modify this file and carry an ADR explaining why. A rule that
is routinely violated is either wrong or unenforced — fix one or the other, do not
leave it standing.
