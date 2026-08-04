# ADR-0011 — Package name and import layout

- **Status:** **Accepted** — confirmed by the operator 2026-08-04 (`STATE.md` B1, now closed)
- **Date:** 2026-08-03
- **Affects:** every import in the project · blocks M2-T01

## Context

The importable package is currently called `src`. That is not a name, it is a directory:
it collides with every other project that does the same, it forces the `pythonpath = . src`
hack in `pytest.ini`, and it cannot be installed or distributed.

The distribution is named `afm-analysis`. That was accurate when the project was an AFM
height-map pipeline. It no longer is: `MicroscopyData` and the SEM/TEM path are in the
code, the frontend exposes three modalities, and the audit treats TEM support as a
first-class defect (D-12). The application being built is "a Linux desktop application
for nanoparticle microscopy analysis" — AFM is one modality among three.

Renaming is cheap now, while the package has no external users, and expensive later —
every import, every document, every notebook, every packaging artifact.

## Decision

- **Import package:** `nanoscope`
- **Distribution name:** `nanoscope`
- **Console script:** `nanoscope`
- **Layout:** `src/`-less, top-level package with the layer subpackages from ADR-0001:
  `nanoscope/{app,core,application,infrastructure,gui,resources}`
- `py.typed` marker at the package root; the package is installed in editable mode, and
  the `pytest.ini` path hack is deleted (M2-T14).
- The old `src` package survives as a thin re-export shim during M2 and is deleted at
  M2-T15, once nothing — including notebooks — imports it.

The AFM-specific science keeps AFM-specific naming *inside* `core/science` (`afm_io`,
`z_flat`, substrate estimation). The package name describes the product; module names
describe the physics.

## Consequences

**Positive**

- The name matches what the product does, before any external user depends on the old one.
- Installable, importable, distributable; no `sys.path` manipulation anywhere.
- `nanoscope.core`, `nanoscope.gui` read clearly and make the layering visible at every
  import site.
- Removes the `src` collision, which is a real problem the moment two projects share a
  virtualenv.

**Negative**

- Every import in the repository changes, plus notebooks, plus `PROJECT_CONTEXT.md`,
  plus the README examples.
- `nanoscope` is not obviously unique on PyPI; if the project is ever published, the
  distribution name may need a suffix even if the import package stays.
- Git history for every moved file becomes harder to follow (mitigated: pure moves in
  their own commits, no content changes).

**Neutral**

- The repository directory name (`AFM-analysis`) does not have to change and is not part
  of this decision.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Keep `src` | Not a package name; blocks installation; keeps the pytest path hack; collides across projects. | Never. |
| `afm_analysis` (match the current distribution) | Locks the name to one of three supported modalities, on a product explicitly generalising beyond AFM. | The operator decides AFM is and remains the product, and SEM/TEM stay secondary. |
| `nanoparticles` | Describes the object of study, not the tool; awkward as a namespace. | — |
| `npanalyzer`, `particlelab`, other coinages | No better than `nanoscope` and less pronounceable. | The operator prefers one. |
| `nanoscopy` | Slightly more precise as a field name, marginally more awkward as an identifier. | Operator preference. |

## Open question for the operator

**Confirm or replace `nanoscope`.** This is the only thing blocking M2-T01, and the cost
of changing the answer rises with every file moved. Any name works technically — the
decision is a product one.

## Compliance

- After M2-T15: `grep -rn "^from src\|^import src"` returns nothing outside `docs/audit/`.
- `pytest.ini` contains no `pythonpath` entry.
- `pip install -e .` followed by `python -c "import nanoscope"` succeeds from any directory.

## References

- `docs/Architecture.md` §3.1
- `docs/TASKS.md` M2-T01, M2-T14, M2-T15
- `docs/STATE.md` B1
