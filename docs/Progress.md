# Progress

Append-only session log. Newest entry on top. Every session adds one entry, finished or not.

**Entry format:** date · milestone · task IDs · what changed · what was learned · what is next.
A session that changes scientific output states the numerical delta explicitly.

---

## 2026-08-04 — M1 · `M1-T08` CI · **verified locally, not yet run on GitHub**

**Task:** M1-T08 (complete, with one caveat below)
**Branch:** `chore/ci`
**Scientific impact:** none. Golden zero drift, in two environments.

### What was added

`.github/workflows/ci.yml` — format → lint → tests+golden → legacy report, on push and
pull request. About four minutes, three of which are the golden.

### The hard part was the environment, and it went the opposite way to expectations

`CURRENT_TASK.md` flagged the risk that `uv sync` would drag torch off the CUDA 11.8 index
and clone SAM2 from GitHub. Tracing the imports first showed something better than a
workaround: **`src/` never imports torch at all.** `ultralytics` and `patched_yolo_infer`
appear only inside two function bodies in `yolo_detector.py`, SAM2 only inside
`segmentation.py`'s call paths, and `import src.pipeline` pulls in nothing heavier than
matplotlib, pandas, scikit-image and scipy.

So CI installs a `ci` dependency group — numpy, scipy, scikit-image, pandas, matplotlib,
headless OpenCV, plus dev — and **nothing else**. No CUDA wheel, no git clone, no weights.
Versions are not repeated in the group; uv resolves every group against one lock, so CI
gets exactly what the dev machine has. That last point is not cosmetic: the golden is
sensitive to numpy/scipy/scikit-image versions, and it was checked explicitly before
anything else.

| | golden baseline | dev `.venv` | what CI installs |
|---|---|---|---|
| numpy | 2.4.4 | 2.4.4 | 2.4.4 |
| scipy | 1.17.1 | 1.17.1 | 1.17.1 |
| scikit-image | 0.26.0 | 0.26.0 | 0.26.0 |

The claim is also asserted in the workflow rather than trusted: a step fails the job if
`torch`, `ultralytics`, `sam2` or `patched_yolo_infer` is importable. A CI run that is
green for the wrong reason is worse than a red one.

### Two traps found by running it instead of writing it

**`uv run` re-syncs the project environment by default.** Every `uv run ruff …` step would
have silently reinstalled the full runtime — torch included — undoing the `--only-group ci`
install one step earlier, and the job would still have gone green. `UV_NO_SYNC: "1"` is set
at job level, and the CPU-only assertion above is what would catch a regression.

**`ruff format --check .` failed on three Markdown files.** Ruff formats Python inside
Markdown code blocks, and `docs/Architecture.md` contains illustrations, not source — a
`Protocol` sketch with `...` bodies and a registry call aligned by hand for reading. A
formatter must not be authoritative over prose examples, so `*.md` is now excluded.
Nothing found this by inspection; it took running the command that CI would run.

### One exclusion, three consumers

`src/` and `preprocess_batch.py` were excluded in `.pre-commit-config.yaml` (M1-T07) and
would have needed excluding again in the workflow — a third place to drift. They now live
in `[tool.ruff] extend-exclude` with `force-exclude = true`, declared once:

- `ruff check .` / `ruff format --check .` → the code we own; blocking in hooks and CI
- `ruff check src preprocess_batch.py --no-force-exclude` → the legacy baseline, on demand

`force-exclude` is what extends the exclusion to paths named explicitly on the command
line, which is what pre-commit passes. Verified afterwards that the hooks still skip `src/`.

### mypy reports a different number in CI, and it is not a bug

**21 errors in CI, 22 locally.** The missing one is
`yolo_detector.py:99 Incompatible types in assignment (list[Any] vs None)` — part of
M3-T18. With `ultralytics` absent, mypy types `YOLO(...)` as `Any`, and assigning `Any` to
a `None`-typed attribute is legal, so the defect becomes invisible.

A type checker's output depends on which third-party packages are installed. Worth
remembering when M2 makes `nanoscope/` strict: the strictness is only as good as the stubs
present. Documented in `Development.md` §4 with both numbers, because a CI summary that
disagrees with `STATE.md` and explains nothing is how people learn to ignore CI.

### Verification

The workflow has **not** run on GitHub — that needs a push, and nothing in this session has
been pushed. What was done instead: the exact command sequence was executed against a
scratch environment built the way CI builds it (`uv sync --only-group ci --locked` into a
separate `UV_PROJECT_ENVIRONMENT`, leaving the dev venv untouched).

| Step, run in the CI-shaped environment | Result |
|---|---|
| `uv sync --only-group ci --locked` | 42 packages, no torch/ultralytics/sam2/patched-yolo-infer |
| CPU-only assertion | passes |
| `ruff format --check .` | clean |
| `ruff check . --no-fix` | **All checks passed** |
| `pytest -q` | 23 passed, 195 s, zero golden drift |
| legacy report | 117 ruff findings, 21 mypy errors |
| **broken parser** (z divisor mutated) | 3 failed → **exit 1** |
| **drifted golden** (`height_nm.mean` +0.1%) | 1 failed, quantity named → **exit 1** |

Both rejection cases are the DoD's real question — a CI that cannot fail is the same
non-gate as a test that cannot fail (M1-T05, M1-T06). Confirmed red on both.

### Decisions recorded

- **No `pre-commit run --all-files` in CI.** Its blocking content already runs directly
  with the same configuration, and `--all-files` is red on the two committed notebooks,
  which are M1-T09's property. CI must not be the thing that forces an unrelated task.
- **No README badge.** `README.md` is stale until M9 (D-24). A green badge on a document
  that misdescribes the project claims health it does not have.
- **mypy is non-blocking**, because `files = ["src"]` and `src/` is the legacy core. M2-T01
  points it at `nanoscope/`, where it is strict and blocking from the first line.

### Learned

- **Read the imports before designing around them.** The expensive mitigation this task
  budgeted for — a CPU torch index, a trimmed install — turned out to be unnecessary
  because the code already isolates its heavy dependencies. The cheapest answer was
  available only after tracing what actually gets imported.
- **A CI step that cannot fail is invisible.** `uv run`'s implicit sync would have produced
  a green job in a wrong environment. The assertion step exists because of it.
- **Tooling has opinions outside its remit.** Ruff reformatting documentation prose is the
  same class of surprise as `fix = true` rewriting the scientific core (M1-T03).

### Next

`M1-T09` — notebooks. Then `M1-T10` (`make check`) closes M1.

**Open, and now the only thing between here and M2: B1, the package name.**

**Also open: this workflow has never executed.** It needs one push to `origin` to be real.

---

## 2026-08-04 — M1 · `M1-T07` Pre-commit — the first mechanism that can refuse

**Task:** M1-T07 (complete)
**Branch:** `chore/pre-commit`
**Scientific impact:** none. No file under `src/` is modified; the golden reports zero
drift. The characterization harness was reformatted — see below, with the proof.

### What was added

`.pre-commit-config.yaml`, nine hooks, and `pre-commit 4.6.1` in the dev group.

**ruff runs as a `repo: local` hook**, calling the project's own `uv run ruff`. The
conventional `astral-sh/ruff-pre-commit` mirror declares a second ruff version in a second
file, and the two drift until local and CI disagree about what counts as a finding.
`pyproject.toml` is now the only place a version is stated.

**`ruff check --no-fix`, never `--fix`.** Formatting is not an opinion (PROJECT_RULES §3:
`ruff format` decides), but a lint autofix rewrites logic — and M1-T03 removed `fix = true`
for exactly this reason. Format rewrites; check reports.

**pytest and mypy are not hooks.** The golden alone is 200 s. A hook that slow is a hook
people bypass with `--no-verify`, and a gate that gets routed around is worse than no gate.
They go to CI (M1-T08).

### Every hook was proven to fire

Nine hooks, each given a deliberately bad staged file:

| Probe | Result |
|---|---|
| 2 MB binary | `check-added-large-files` **refused** |
| unformatted Python | `ruff format` rewrote it, commit **aborted** |
| unused import | `ruff check` **refused** |
| trailing whitespace / no final newline | both fixers **rewrote**, commit **aborted** |
| broken YAML, broken TOML | both **refused** |
| notebook with outputs | `nbstripout` **stripped** them, commit **aborted** |

An accident along the way: the first large-file probe used a `.pt` file and nothing
happened, because M1-T01's `.gitignore` had already excluded it. The hook is the second
line of defence, not the first.

### What `--all-files` revealed — the reason this task nearly shipped a bug

The sweep modified **`src/measure.py`, `src/preprocess.py`, `src/visualization.py` and
`preprocess_batch.py`**. The ruff hooks were excluded from `^src/`, but two things were not
caught by that:

- `end-of-file-fixer` and `trailing-whitespace` had **no exclusion at all** and trimmed
  inside the scientific core;
- `preprocess_batch.py` lives at the repository root, not under `src/`, so `ruff format`
  reformatted it — it is core code that the path-based exclusion simply missed.

Everything was reverted and the config now uses one named exclusion, `^(src/|preprocess_batch\.py)`,
applied to every hook that **rewrites** a file. Hooks that only **refuse** — large files,
merge conflicts, YAML/TOML — still apply everywhere, `src/` included. Nothing is exempt
from being stopped; the core is only exempt from being edited.

The posture is deliberate and matches mypy's from M1-T04: the core is reported, not
silenced, and not rewritten. Two reasons, neither of them taste. `ruff check` reports 109
findings in `src/`, so a blocking hook there would make every commit that touches the core
impossible — M2 is sixteen such tasks, and the gate would be bypassed on day one. And
PROJECT_RULES §4.1 forbids rewriting the science to make the architecture prettier: a
whitespace trim riding inside an M2 relocation commit is noise in the one diff that has to
be readable as a pure move.

### The characterization harness was cleaned, deliberately

`--all-files` also flagged 8 ruff findings and formatting in `tests/characterization/`.
Those were applied by hand rather than reverted, because a gate that is red on the day it
arrives gets ignored. All eight are behaviour-identical — `int(len(x))` → `len(x)`, two
dead `noqa: BLE001` directives (the `S`/`BLE` families are not selected), import order, and
line joins. **The golden was run afterwards and reports zero drift**, which is the only
argument that counts for a file that generates the baseline.

### Damage report — an uncommitted file was rewritten

`pre-commit run --all-files` ignores the index and rewrites the working tree. The tree held
an uncommitted `project.md` from before this session; the sweep restored its missing final
newline, and the file is now byte-identical to `HEAD`. Nothing was lost beyond that one
newline — 11752 bytes before, 11753 after, no textual difference — but the hazard is real
and is now a warning in `docs/Development.md` §4: commit or stash before running
`--all-files`.

### Measurements

| | |
|---|---|
| Hooks configured / proven to fire | 9 / 9 |
| `pytest` | 23 passed, 188 s |
| Characterization golden | zero drift, after the harness reformat |
| Files under `src/` modified by this task | **0** |
| `--all-files` still failing on | the two committed notebooks (M1-T09) and one archived doc — knowingly, both are other tasks' property |

### Learned

- **A path-based exclusion is only as good as the paths.** `^src/` looks like "the
  scientific core" and is not: `preprocess_batch.py` sits at the root and imports it. The
  sweep is what showed this; a config review would not have.
- **Hooks that rewrite and hooks that refuse need different scopes.** Conflating them
  either blocks legitimate commits or edits code nobody asked to touch. Splitting the two
  made the whole configuration obvious.
- **`--all-files` is not a dry run.** It edited a file that was not staged, not committed,
  and not part of this task.
- The `.gitignore` from M1-T01 already stopped the model-weight probe before pre-commit saw
  it. Layered defences are working as intended, and worth remembering when reading a green
  hook run: it may be green because something earlier said no.

### Next

`M1-T08` — CI. The slow half of the gate that pre-commit deliberately refuses to run:
`pytest` including the golden, plus ruff and mypy reporting on `src/` without blocking.

**B1 remains the only thing blocking M2.**

---

## 2026-08-04 — M1 · `M1-T06` A real test for the SPM parser · **the suite is green**

**Task:** M1-T06 (complete)
**Branch:** `test/spm-io`
**Scientific impact:** none — `src/` is not edited. The golden reports zero drift.

### What was there

Eleven lines that tested nothing: no assertion, `z` assigned and never read, `ImportError`
caught for `pyfmreader` (a package this project does not depend on — the parser is
hand-written) while the actual failure is `FileNotFoundError`, and a read of `data/5.011`,
which is git-ignored and absent from any clean checkout. It failed on every machine, and
had the file been present it would have passed regardless of what the parser returned.

### What replaces it

`tests/unit/test_afm_io.py` — **22 tests**, no binary fixture, no `data/`, no network.

The fixture is a synthetic Nanoscope SPM byte stream built in the test module: preamble,
a decoy image block, the Height block, `0x1A`, padding to the declared data offset, then
an `int16`/`int32` payload. Field names and formats were taken from a **real** local file
(`data/pvp8k/2-6-dmfa-pvp.039` — read, not committed), including the two details that
matter: Nanoscope writes micrometres as `~m`, and every header carries a second
sensitivity, `@Sens. ZsensSens`, thirty times the real one.

| Group | Covers |
|---|---|
| Round trip | shape, `float32`, `[y, x]` orientation, values (non-square 6×4, so a transpose cannot survive), Height-block selection over a decoy, `int32` when `Bytes/pixel != 2` |
| Calibration | `pixel_size_nm == scan_size_nm / samps`, the full LSB → volts → nm chain, and `~m` / `um` / `µm` / `nm` conversion |
| Failure modes | missing file · no Ciao blocks · missing header field · no Z scale · no Zsens · truncated payload · no `Scan Size` (M3-T17) · unsupported format |
| Other entry points | `fmt="npy"` with and without metadata; `load_microscopy_image` greyscale round trip, unknown scale, missing file |

### The suite was tested, not just written

A test suite that has never failed is a hypothesis. Four mutations of `src/afm_io.py`, run
and reverted:

| Mutation | Result |
|---|---|
| `pixel_size_nm = scan_size_nm / lines` instead of `/ samps` | **5 failed** |
| Z scale divisor `65536 → 32768` | **4 failed** |
| Height-block selection replaced by "take the first block" | **13 failed** |
| Zsens regex loosened to `Zsens\w*`, so it also matches `ZsensSens` | **survived** |

The fourth is the one worth recording. The decoy `ZsensSens` line was only being written
when the correct `Zsens` line was also present and earlier in the file, so `re.search`
found the right one either way and the test proved nothing. The real hazard is a header
that has `ZsensSens` but no `Zsens`: a loosened pattern would then silently scale every
height in the scan by ~30 and raise nothing. The fixture now always carries the decoy, and
`test_spm_without_zsens_is_rejected_and_zsenssens_is_not_a_substitute` kills the mutant.

Written and passed, that test was decoration. Only the mutation showed it.

### New defect found — M3-T20

`load_afm(fmt="npy")` fabricates a physical scale: `pixel_size_nm or 1.0` and
`scan_size_nm or float(z.shape[0])`. PROJECT_RULES §3 and D-07 both say an unknown scale is
`None` — never a stand-in. So every downstream `_nm` on that path is a pixel count wearing
nanometre units; the row count is used as a length in nanometres, which is not even
dimensionally a size; and because it is written with `or`, a caller who explicitly passes
`0.0` is overruled too. Not in the audit, not previously filed → **M3-T20**, high.

Both this and M3-T17 are pinned by assertions that name the task, so the fix flips a
documented expectation instead of breaking a surprise.

### Measurements

| | |
|---|---|
| `pytest` | **23 passed**, 200 s — first green run in the project's history |
| `pytest -m "not slow"` | 22 passed, **0.88 s** |
| Characterization golden | zero drift |
| mypy | 22 errors, unchanged — `files = ["src"]`, tests are not checked |
| ruff check / format on the new file | clean |
| Binary fixtures added | none |

### Learned

- **The parser is more testable than it looks.** It needs six header fields and two regex
  matches; a faithful fixture is ~60 lines. The reason it had no tests was not difficulty.
- **Deriving the fixture from a real file paid for itself immediately.** `~m` for
  micrometres and the `ZsensSens` twin are not things one invents at a desk, and both are
  now covered.
- **Mutation testing found the one worthless test out of 23.** Cheap — four edits and four
  runs — and it is the only reason the Zsens guard is real. Worth repeating whenever a test
  claims to defend a subtle regex or a unit conversion.
- The fast loop is 0.88 s, of which **1.4 s of import cost is D-18** — `import src.afm_io`
  pulls 1209 modules through `src/__init__.py`. It is inside pytest's startup rather than
  the test time, but it is the same defect, and M2-T09 removes it.

### Next

`M1-T07` — pre-commit hooks. With the suite green, the gate can start refusing bad commits
instead of reporting them afterwards.

**B1 remains the only thing blocking M2.**

---

## 2026-08-04 — M1 · `M1-T05` The golden runs under pytest

**Task:** M1-T05 (complete)
**Branch:** `chore/golden-in-pytest`
**Scientific impact:** none — no golden value changed, no numerical code touched.
`capture.py`'s comparison, tolerances and digests are byte-for-byte the same; the CLI
prints the same line it printed before (`characterization baseline stable (9 groups)`).

### What changed

- **One seam in `capture.py`**: `diff_against_golden() -> list[str]` — read the golden,
  `build_all()`, `compare()`, return the path-addressed diff. `main()` now calls it and
  keeps sole ownership of printing and exit codes. Nothing else in the file moved.
- **`tests/characterization/test_golden.py`** — one `@pytest.mark.slow` test that asserts
  the diff is empty and puts it in the assertion message. It reimplements nothing; if the
  test and the CLI ever disagreed it would be because they share a code path, and they do.
- **`pytest.ini` deleted, configuration folded into `pyproject.toml`** (scope item 7).
  This was the one open decision in the task, and the deciding fact is not tidiness: while
  a `pytest.ini` exists pytest ignores `[tool.pytest.ini_options]` **entirely and
  silently**. Two files that can shadow each other is exactly the failure mode this task
  exists to remove. The `pythonpath = [".", "src"]` hack moved across unchanged and still
  dies in M2-T14.
- `docs/Development.md` §4 and §5 document both invocations.

### The proof that matters

A test that cannot fail is not a safety net, so the negative case was run rather than
assumed. `afm_flat_monodisperse…detect_particles_p20.n_blobs` was edited 24 → 23 in the
golden file:

```
E   AssertionError: CHARACTERIZATION DRIFT: 1 difference(s)
E       afm_flat_monodisperse.log_detection.detect_particles_p20.n_blobs: 23 -> 24
```

Red, one line, the quantity named with both values. The golden was then restored;
`git diff` on `baseline.json` is empty.

### Measurements

| | |
|---|---|
| `pytest tests/characterization/test_golden.py` | **passed**, 192 s |
| `pytest -m "not slow"` | 1.4 s, golden deselected, `test_io.py` fails as expected (M1-T06) |
| `python tests/characterization/capture.py` | unchanged output, exit 0 |
| Marker warnings | none |
| ruff on the new file | clean, formatted |

### Learned

- **The task estimated ~100 s; it is 192 s.** The figure in `Development.md` was inherited
  and never measured. Corrected there. It matters: this is the number that decides whether
  people keep running the full suite, and `-m "not slow"` is the answer to it.
- **`pytest.ini` + `[tool.pytest.ini_options]` is a silent-override trap.** Had the marker
  been registered in `pyproject.toml` while `pytest.ini` still existed, the registration
  would have done nothing and the warning would have stayed — with no error to explain it.
- **The harness was already test-shaped.** `build_all()` and `compare()` were pure; only
  `main()` mixed in printing. One extracted function was the whole job — no restructuring,
  hence no risk to the numbers.
- The two `RuntimeWarning`s pytest now surfaces (`Mean of empty slice`, `Degrees of freedom
  <= 0`) are not new. They come from the degenerate-input phantoms and always went to
  stderr; the CLI just made them easy to overlook. They are characterized behaviour.

### Next

`M1-T06` — replace `tests/test_io.py`. It is the only thing keeping `pytest` red: no
assertions, catches `ImportError` while the real failure is `FileNotFoundError`, and it
reads `data/5.011`, a path that does not exist in a clean checkout.

**M2 is no longer blocked by the safety net** — the golden is mechanically enforced. It is
still blocked by **B1**, the package name.

---

## 2026-08-04 — M1 · `M1-T04` Mypy configuration

**Task:** M1-T04 (complete)
**Branch:** `chore/mypy-config`
**Scientific impact:** none — configuration only, no source file edited

### The 30 errors, classified before writing any configuration

The rule for this task was that configuration must not silence a real bug. So all 30
default-run errors were read against the source first.

| Class | Count | Disposition |
|---|---:|---|
| Missing third-party stubs | 8 | silenced per module — pandas, scipy, patched_yolo_infer, ultralytics |
| **Static confirmation of known audit defects** | **13** | kept visible; already have M3 tasks |
| **Real typing defects not previously filed** | **5** | kept visible; **filed as M3-T17…T19** |
| Stub strictness / known stub | 4 | kept visible, harmless |

**The 13 that confirm the audit.** mypy independently reproduces, statically, defects the
Phase 0 audit found by execution:

| Error | Defect |
|---|---|
| `preprocess.py:202  Cannot determine type of "opening_radius"` | **D-01** — the critical `UnboundLocalError`. The manual-radius branch never assigns it; mypy sees the same missing assignment the runtime does. |
| `preprocess.py:149,158  return float, expected int` + `:184 arg-type float→int` | **D-10** — `estimate_rough_radius` is annotated `-> int` and returns a float, which reaches `disk()` and produces even-sized structuring elements. The whole chain is visible in three errors. |
| `types.py:63  tuple[Never, ...] vs tuple[int, int, int, int]` | **D-16** — `bbox` defaults to `()` against a four-element annotation. |
| `preprocess.py:164`, `log_detector.py:125,257`, `pipeline.py:52` | **D-07** — implicit `Optional` and the unknown-pixel-scale contract. |
| `pipeline.py:94 ×2, :110  ndarray \| None where ndarray expected` | the SEM/TEM path, where `z_flat` is `None` by construction. |
| `afm_io.py:100  returns tuple, annotated ndarray` | **D-02** — the return-convention change that silently broke `preprocess_batch.py`. |

A configured type checker would have caught the project's single critical defect before
it was ever committed.

**The 5 that are new.** Filed as tasks, not suppressed:

- **`afm_io.py:98` — new defect, not in the audit.** When the header carries no
  `Scan Size:` field the code sets `scan_size_nm = None` and then immediately evaluates
  `pixel_size_nm = scan_size_nm / samps` → `TypeError`. The `else` branch exists
  specifically to handle that header, and it crashes on the next line. → **M3-T17**
- `yolo_detector.py:50,87,99` — `_last_result` is initialised to `None`, so its inferred
  type is `None`; `.filtered_boxes` is then accessed unguarded. → **M3-T18**
- `log_detector.py:111,116` — `responses` is annotated `list[float]` and then rebound to
  an ndarray before `.min()`/`.max()`. Works at runtime, wrong as a contract. → **M3-T19**

### Configuration

- `[tool.mypy]`: `python_version = "3.12"`, `files = ["src"]`, `warn_unused_configs`,
  `warn_redundant_casts`, `warn_unused_ignores`
- **`nanoscope.*` is strict from its first line** — `disallow_untyped_defs`,
  `disallow_incomplete_defs`, `disallow_untyped_calls`, `disallow_any_generics`,
  `check_untyped_defs`, `no_implicit_optional`, `warn_return_any`, `strict_equality`.
  Retrofitting strictness later is far more expensive than starting with it.
- **`src/` posture: checked, not silenced.** No `ignore_errors`. It carries 22 errors,
  13 of which are the most valuable output this tool has produced; hiding them to make a
  number green would be the opposite of the point. The package is deleted in M2-T15, so
  the errors are a documented baseline, and CI reports them without blocking (M1-T08).
- `ignore_missing_imports` scoped **per module**, never globally — a blanket setting would
  also hide a typo in a first-party import.

`mypy` now runs with no command-line flags: **22 errors in 7 files** (30 minus the 8
stub gaps). It emits one note — `unused section(s): module = ['nanoscope.*']` — which is
deliberate: it is a visible reminder that M2-T01 has not happened yet, and it disappears
the moment the package exists. Verified non-fatal: mypy exits 0 on a clean file with that
note present.

### Considered and rejected

Installing `pandas-stubs` and `scipy-stubs` instead of silencing those imports. It would
give real coverage, but pandas-stubs against pandas 2.x typically produces a fresh wave of
errors in code that is scheduled for deletion in M2. Revisit when `nanoscope` exists —
backlog **B-057**.

### Next

`M1-T05` — wire the characterization golden into `pytest`. It is the only check that
passes today and the only protection M2 has, and it currently runs only when someone
remembers to type the command.

---

## 2026-08-03 — M1 · `M1-T03` Ruff configuration repair

**Task:** M1-T03 (complete)
**Branch:** `chore/ruff-config`
**Scientific impact:** none — `capture.py` reports `characterization baseline stable (9 groups)`

### Done

- **Removed `fix = true` / `show-fixes`.** `ruff check .` was rewriting source files as a
  side effect of being asked a question — 66 automatic edits to the scientific core, from
  a command the documentation told people to run. Fixing is now explicit: `ruff check --fix`.
- Moved `select` / `ignore` under `[tool.ruff.lint]`; the deprecation warning is gone
  (stderr is now empty).
- `target-version` `py311` → `py312`; the project requires `>=3.12`.
- `known-first-party` `["your_package_name"]` → `["src"]` (unedited template value; it
  becomes the real package in M2-T01).
- `classmethod-decorators`: dropped `pydantic.validator` — pydantic is not a dependency.
- `per-file-ignores`: dropped `S101`; the `S` (bandit) family is not selected, so the
  entry was dead configuration. Backlog **B-056**.
- Excluded `*.ipynb` from lint. Notebooks are experiments, not interfaces
  (PROJECT_RULES §7); their 68 findings are import order and prints in exploratory cells.
  Notebook hygiene is M1-T09.

### Verification

| Check | Result |
|---|---|
| Config deprecation warning | gone — stderr empty |
| `ruff check .` modifies the tree | **no** — `git diff --exit-code` clean after every run |
| `src/` findings before vs after | **identical** — `--statistics` diff is empty |
| Total findings | 196 → **128** (the 68 excluded are all notebooks) |
| `ruff format --check .` | runs; 18 files would be reformatted (not fixed here — M2) |
| Characterization | zero drift |

### Correction to the M1-T02 baseline

The `src/` figure recorded in M1-T02 was **109, not 108**. I produced the 108 by summing
a `--statistics` listing that I had truncated with `head -20`, dropping the last row
(`W291 trailing-whitespace`, 1). The commit message of `13857e5`, and `STATE.md` /
`TASKS.md` before this entry, carry the wrong number.

The distribution is otherwise unchanged, and the invariant this task was checked against
still holds: the configuration repair changed **nothing** about what is reported — the
before/after statistics diff is empty. Corrected everywhere in the living documents;
the commit message of `13857e5` is history and stays as written.

Burn-down target for M2 is therefore **109 findings in `src/`**, of which 44 are the
ambiguous-unicode signature of the Russian text (D-22) and 13 are `print` (D-23).

### Next

`M1-T04` — mypy configuration. 30 errors today with default settings; the task is to
choose strictness for new code and a baseline exclusion for `src/` until M2 lands, not to
fix the errors.

---

## 2026-08-03 — M1 · `M1-T02` Dev dependencies and quality baseline

**Task:** M1-T02 (complete)
**Branch:** `chore/dev-dependencies`
**Scientific impact:** none — `capture.py` reports `characterization baseline stable (9 groups)`
after the environment change

### Done

- `[dependency-groups] dev` added to `pyproject.toml`: pytest, pytest-cov, ruff, mypy
- `uv sync`; `uv.lock` gained **only** tooling packages — no runtime version moved.
  Verified against the golden `_meta` pins: torch 2.7.1+cu118, NumPy 2.4.4, SciPy 1.17.1,
  scikit-image 0.26.0 — all unchanged, which is why the golden is still stable

| Tool | Version |
|---|---|
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |
| ruff | 0.16.1 |
| mypy | 2.3.0 |

### Baseline — the M2 burn-down target

Measured, nothing fixed. These are the numbers M2 has to drive to zero.

**ruff — 196 findings total, 109 in `src/`** (66 auto-fixable)
*(corrected in the M1-T03 entry above; this session recorded 108 from a truncated listing)*

| Rule | src/ | What it is |
|---|---:|---|
| RUF002/003/001 | **44** | ambiguous unicode in docstrings, comments, strings — this is the Russian text of **D-22**, found mechanically |
| T201 | **13** | `print` in library code — **exactly the 13 the audit counted by hand** |
| F401 | 11 | unused imports |
| I001 | 11 | unsorted imports |
| W293/W292/W291 | 10 | whitespace |
| RET504/505 | 7 | unnecessary assign / superfluous else |
| RUF046 | 2 | unnecessary `int()` cast — **adjacent to D-10**, the opening-radius rounding defect |
| RUF013 | 2 | implicit `Optional` — the unknown-scale contract (**D-07**) |
| A005 | 1 | `src/types.py` shadows the stdlib `types` module — a real M2-T02 constraint |
| others | 8 | B007, C408, N806, PIE790, RUF022, SIM108, UP037 ×2 |

The remaining 88 findings are in notebooks, `preprocess_batch.py` and
`tests/characterization/capture.py`.

**mypy — 30 errors in 9 files** (default settings, no configuration yet)

| Code | Count |
|---|---:|
| import-untyped | 8 |
| assignment | 7 |
| arg-type | 6 |
| return-value | 3 |
| attr-defined | 3 |
| has-type, empty-body, call-overload | 3 |

The most interesting one is static confirmation of a known runtime defect:

```
src/pipeline.py:110: error: Argument 4 to "run_sam2_from_blobs" has incompatible type
"ndarray[...] | None"; expected "ndarray[...]"
```

That is the SEM/TEM path, where `z_flat` is `None` by construction (`pipeline.py:53`).
A type checker would have caught it before it ever ran.

**pytest — 1 test, 1 failed**

```
FAILED tests/test_io.py::test_load_spm - FileNotFoundError: 'data/5.011'
```

Correction to the prediction in `CURRENT_TASK.md`: the test does not pass vacuously, it
**fails**. It catches `ImportError` while `load_afm` raises `FileNotFoundError` (audit
D-20), and the path is absent from a clean checkout. The suite has been red the whole
time; nobody could see it because pytest was never installed. M1-T06 replaces it.

### Side effect worth knowing about

`uv sync` uninstalled three packages that were in the environment but not in `uv.lock`:
`clip` (from the ultralytics CLIP repo), `ftfy`, `regex`. They were installed outside uv,
so `uv sync` removed them to match the lock — expected behaviour, but not something I
intended.

Nothing under `src/` imports them; they are needed only for **YOLO-World** models, and
`checkpoints/yolov8s-world.pt` is such a model (it is not the configured default —
`PipelineConfig.yolo_model_path` points at `best12x.pt`). If YOLO-World is wanted, the
fix is to declare it as a real dependency rather than let it be installed ad hoc:

```bash
uv add "clip @ git+https://github.com/ultralytics/CLIP.git"
```

Recorded as backlog **B-055**.

### Not done, deliberately

No finding was fixed. Repairing the ruff configuration is M1-T03, the mypy configuration
is M1-T04, and the 109 `src/` findings are M2 work — under the protection of the golden
file, not before it.

### Next

`M1-T03` — repair the ruff configuration. It currently emits a deprecation warning for
top-level `select`/`ignore`, targets `py311` on a 3.12 project, still carries
`known-first-party = ["your_package_name"]`, and — most importantly — sets `fix = true`,
so `ruff check .` rewrites source files as a side effect of being asked a question.

---

## 2026-08-03 — M1 · `M1-T01` Repository hygiene

**Task:** M1-T01 (complete), M1-T11 (complete — absorbed)
**Branch:** `chore/repo-hygiene`
**Scientific impact:** none — `capture.py` reports `characterization baseline stable (9 groups)`

### Result

| Metric | Before | After |
|---|---:|---:|
| Tracked files | 2 877 | **77** |
| Tracked under `frontend/node_modules` | 2 800 | **0** |
| Tracked model weights | 1 (`yolov8s-world.pt`, 26 MB, staged) | **0** |
| Largest tracked non-notebook file | 6.5 MB | 3.2 MB (README figure) |

### Done

- `git rm -r --cached frontend/node_modules` — 2 800 files untracked, all still on disk
- `git rm --cached yolov8s-world.pt` — the 26 MB blob (`2fa1b38`) was staged for addition
  and would have entered history on the next `git commit` without a pathspec. Removed
  from the index before that happened; the file is not on disk and the four real
  checkpoints under `checkpoints/` are untouched
- `.gitignore` rewritten: added `node_modules/`, `output/`, `*.pt`, `*.pth`, `*.onnx`,
  `*.safetensors`, `*.zip`, `*.tar.gz`, `build/`, `dist/`, `*.egg-info/`, `.mypy_cache/`,
  `.coverage`, `htmlcov/`; grouped and commented by rationale
- `.claude/settings.json` now **tracked** — agent configuration is shared (PROJECT_RULES §7).
  `.claude/settings.local.json` (per-machine permissions) stays ignored. This completes
  the `.gitignore` edit that was already sitting uncommitted in the working tree
- **Junk removed from disk:**
  - `.zip` — a 22-byte *empty* zip archive, tracked since February
  - `__pycache__/` × 4, including `.pyc` files for modules that no longer exist
    (`sam2_pipeline`, `config`, `detection`) and bytecode from CPython 3.14 while the
    venv is 3.12 — stale in two independent ways
  - `.pytest_cache/`, `.ruff_cache/`
  - `output/`, `notebooks/` — both empty directories
  - root `package-lock.json` — an empty stray from an accidental `npm install` at the
    repository root (the real one is `frontend/package-lock.json`)
- `plan.md` → `docs/archive/plan-frontend-react-client.md`, un-ignored and now tracked,
  with an ARCHIVED header pointing at ADR-0007. It was gitignored, so the only record of
  the intended HTTP contract was unshareable. Path references in ADR-0007 updated
  (editorial only — the decision is unchanged)

### Deviation from the stated Definition of Done

"Largest tracked file, excluding the pre-existing notebooks, < 1 MB" is **not met**:
`images/yolo_sam2_comparison.png` (3.2 MB) and `images/log.png` (3.0 MB) are README
figures. They are legitimate content, not junk, and untracking them would break the
README. Recorded as backlog **B-054** (optimise figures) rather than silently ignored.

The notebooks (6.5 MB + 2.2 MB, committed with outputs) are untouched — that is M1-T09.

### Not done, deliberately

History still carries the 78 MB: `git rm --cached` stops the growth, it does not shrink
`.git` (still 81 MB). Rewriting history invalidates every clone and needs the operator's
approval — backlog **B-040**.

### Next

`M1-T02` — declare `pytest`, `pytest-cov`, `ruff`, `mypy` as dev dependencies. None of
them is installed today, so the quality gate does not exist yet; the characterization
runner is currently the only working check.

---

## 2026-08-03 — M0 · Engineering foundation

**Tasks:** M0-T01 … M0-T08 (all complete)
**Branch:** `docs/engineering-infrastructure`
**Base:** `11e0ecc` (frontend init)
**Code changed:** none — documentation only, by design

### Done

- Read `systempromt.md`, `PROJECT_CONTEXT.md`, `README.md`, plus `project.md`, `plan.md`,
  the Phase 0 audit and the characterization baseline
- Analysed the repository directly: 12 modules / 2 021 LOC under `src/`, 13 frontend
  source files, 8 characterization phantoms, 2 854 tracked files
- Recorded 7 strengths and 16 weaknesses with measured evidence → `Architecture.md` §2
- Defined the target Clean Architecture: `app` / `core` / `application` / `infrastructure` /
  `gui` / `resources`, with an enforced dependency rule and a layer-contract table
- Wrote the constitution → `PROJECT_RULES.md`
- Broke the project into 10 milestones (M0–M9) with exit criteria → `Roadmap.md`
- Broke the milestones into 110 tasks → `TASKS.md`
- Wrote 11 ADRs (0001–0011) → `docs/ADR/`
- Established the state protocol → `STATE.md`, `CURRENT_TASK.md`, this file
- Selected `M1-T01` as the first task

### Learned

- **The starting position is better than a greenfield.** A completed, *reproduced* Phase 0
  audit and a committed golden baseline over 8 seeded phantoms already exist. That changes
  the strategy: the domain can be moved aggressively, because drift is detectable to
  `rtol=1e-6`.
- **The stack pivoted.** The previous direction was React + a FastAPI backend that was
  never written; the target is a Qt6 desktop application. The React client is the only
  work made obsolete by the pivot, and it is parked rather than deleted (ADR-0007).
- **The domain layer is genuinely worth preserving.** A modality-neutral `Detection`, a
  `BaseDetector` ABC, lazily imported SAM2 and a deliberate dependency root mean the
  Clean Architecture target is an extraction, not a rewrite.
- **Two problems are urgent for different reasons.** `node_modules` (98% of tracked files)
  makes review impossible; the staged `yolov8s-world.pt` has a closing window before it
  is permanently in history. Both are M1-T01.
- **Structure must precede correctness.** Fixing D-03 or D-04 today would change numbers
  inside a codebase with 5 import cycles and no test gate. M2 before M3 is not
  bureaucracy — it is the only way the deltas stay attributable.

### Open questions raised

B1 package name · B2 `min_size_nm` semantics · B3 detection polarity · B4 opening-radius
rounding · B5 fate of `frontend/` and the notebooks · B6 real sample data in git.
Full text in `STATE.md`. None blocks M1.

### Next

Execute `M1-T01` on branch `chore/repo-hygiene`: untrack `frontend/node_modules` and
`yolov8s-world.pt`, rewrite `.gitignore`. See `CURRENT_TASK.md`.

---

## Before 2026-08-03 — inherited context

Not a session log; recorded so the history is not lost.

| When | What |
|---|---|
| 2026-07-28 | Phase 0 audit: 24 defects reproduced by execution, 5 import cycles, 10 dead functions → `docs/audit/2026-07-28-baseline-audit.md` |
| 2026-07-28 | Characterization baseline: 8 seeded phantoms, golden file at `rtol=1e-6` → `tests/characterization/`, `docs/audit/characterization-baseline.md` |
| `11e0ecc` | React + Vite frontend scaffolded against an unimplemented `/analyze` backend |
| `e8caf25` | `afm_io` reworked to return `AFMRawData` (silently broke `preprocess_batch.py` — D-02) |
| `cd360aa` | Generalisation to SEM/TEM: `MicroscopyData`, `load_microscopy_image` |
| `f1cf175` | `types.py` and `pipeline.py` introduced — the first deliberate layering |
| `0ef8c50` | Detection refactored to `BaseDetector` + `LogDetector` / `YoloDetector` |
| earlier | SAM2 integration, tiled YOLO, LoG baseline, morphological substrate estimation |
