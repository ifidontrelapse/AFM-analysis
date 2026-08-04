# CURRENT TASK

**ID:** `M1-T06`
**Title:** Replace `tests/test_io.py`
**Milestone:** M1 — Repository hygiene & quality gates
**Status:** selected — not started
**Branch to use:** `test/spm-io`
**Estimated size:** M
**Risk to scientific output:** none — a new test; `src/afm_io.py` is not edited
**Selected:** 2026-08-04

---

## Why this task is next

`pytest` is now wired to something real (M1-T05), and exactly one file keeps it red.

`tests/test_io.py` is the whole of the repository's unit testing, and it tests nothing.
Eleven lines, and every one of them is a problem:

```python
def test_load_spm():
    from src.afm_io import load_afm
    try:
        z = load_afm("data/5.011", fmt="spm")
    except ImportError:
        print("pyfmreader not installed, skipping .spm test.")
        return
```

- **No assertion.** Even on the happy path it asserts nothing about shape, dtype, units
  or scale. `z` is assigned and never read (ruff `F841`).
- **It catches the wrong exception.** The real failure is `FileNotFoundError`; `ImportError`
  is caught for `pyfmreader`, a package this project does not depend on — the parser is
  hand-written in `afm_io.py`.
- **It reads `data/5.011`.** `data/` is git-ignored (628 local scans, B6). On a clean
  checkout the file does not exist, so the test cannot pass for anyone but its author.
- **A `try/except/return` "skip"** hides failure instead of reporting it. `pytest.skip()`
  exists for this.

The result is a test that fails on every machine and, if the file *were* present, would
pass no matter what the parser returned.

The SPM parser is also the least-covered risky code in the project: it is hand-written
binary header parsing, it owns the `scan_size_nm → pixel_size_nm` calibration that every
physical unit downstream depends on, and mypy just found a crash in its no-`Scan Size`
fallback (**M3-T17**). It deserves a real test more than anything else in `src/`.

---

## Scope

**In scope**

1. Delete `tests/test_io.py`; replace it with `tests/unit/test_afm_io.py`
2. **Generate the fixture, do not commit one.** Write a minimal synthetic Nanoscope SPM
   byte stream (header + `int16` payload) in the test module, following
   `tests/characterization/phantoms.py` — deterministic, no binary in git
   (PROJECT_RULES §7), no dependency on `data/`
3. Round-trip: build a known height field → serialise → `load_afm(..., fmt="spm")` →
   assert shape, dtype, `[y, x]` orientation, `scan_size_nm`, `pixel_size_nm`, and the
   height values within the parser's quantisation
4. Assert the **calibration arithmetic** explicitly: `pixel_size_nm == scan_size_nm / samps`
   for a known header, because every `_nm` value in the project is derived from it
5. Characterize the failure modes as tests, not as prose: missing file, truncated payload,
   header without `Scan Size:` (**M3-T17** — assert today's `TypeError` and reference the
   task, so the fix flips a documented assertion instead of a surprise)
6. Cover the SEM/TEM loader (`load_microscopy_image`) with a generated PNG/TIFF if it is
   cheap; if it needs real files, say so and leave it to M3-T16
7. Create `tests/unit/` — the layout `docs/Architecture.md` §3.1 already specifies

**Out of scope**

- Fixing anything in `src/afm_io.py`. The parser's defects (D-02, M3-T17) are M3 work with
  their own ADRs. This task **records** behaviour; it does not change it.
- Real `.spm` files in git (B6 — operator decision)
- `tests/integration/`, `tests/gui/` (later milestones)
- Coverage reporting (pytest-cov is installed, unwired — separate task)

---

## Definition of done

- [ ] `pytest` is **green** — this is the milestone's headline result
- [ ] `pytest -m "not slow"` green and under ~2 s
- [ ] No binary fixture added; `git status` clean apart from the intended files
- [ ] Each new test fails if the parser's behaviour changes — verified by perturbation,
      as in M1-T05
- [ ] The tests pass on a clean checkout with no `data/` directory
- [ ] M3-T17's crash is pinned by an assertion that names the task
- [ ] `docs/STATE.md`, `docs/Progress.md`, `docs/TASKS.md` updated
- [ ] Commit: `M1-T06: replace the SPM I/O test with a synthetic round-trip`

---

## Plan

1. Branch `test/spm-io`
2. Read `src/afm_io.py` end to end — the header fields it requires, the byte layout, the
   `AFMRawData` return convention introduced in `e8caf25` (D-02)
3. Write the synthetic-SPM builder; confirm the parser accepts it
4. Write the round-trip and calibration assertions, then the failure-mode tests
5. Delete `tests/test_io.py`; move the new file to `tests/unit/`
6. Perturb → red → revert, for at least the calibration assertion
7. Run the full gate, including the golden (it must stay at zero drift — nothing in `src/`
   was touched, so any drift means the test module has an import side effect)
8. Update the docs; commit; advance `CURRENT_TASK.md` to `M1-T07`

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **The synthetic file does not represent real Nanoscope output**, so the test passes while real scans break | Build the header from what `_read_nanoscope_z` actually parses, and cross-check the field names against one real file in local `data/` — read it, do not commit it. Note in the test docstring which real header it was derived from. |
| Writing a fixture builder is a rabbit hole | Timebox. The parser needs a handful of header keys; if the builder exceeds ~60 lines, the parser is the problem — record that finding and test the header parsing directly instead. |
| Pinning M3-T17's `TypeError` looks like blessing a crash | The assertion carries the task ID and a comment saying it is expected to flip. That is the point of characterization: the fix must be visible as a test change. |
| A test import pulls in matplotlib/torch and slows the fast loop | `import src.types` loads 1179 modules (D-18). Import `src.afm_io` directly, never the package root, and check the fast run stays under ~2 s. |

---

## Notes for the next session

After T06 the suite is green and M1 is half done. Remaining: T07 (pre-commit), T08 (CI),
T09 (notebooks), T10 (`make check`).

**B1 is still unanswered and is now the only thing blocking M2.** The safety net that M2
depended on became mechanical in M1-T05; the package name is the remaining gate. It should
be answered before M1 closes.
