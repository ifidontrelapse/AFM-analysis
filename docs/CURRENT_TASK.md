# CURRENT TASK

**ID:** `M3-T17`
**Title:** A header without a scan size parses; it does not divide by `None`
**Milestone:** M3 — Numerical correctness, tenth task
**Defect:** **M3-T17** (high), found by mypy in M1-T04 · **ADR:** **ADR-0026**
**Branch:** `sci/spm-header-without-scan-size` (stacked on `sci/npy-no-invented-scale`)
**Status:** planned — no code written yet.

---

## Why this task is next

It is the third and last face of D-07, and the only one left where the unknown-scale state is
still a crash. ADR-0025 gave that state a meaning everywhere downstream one commit ago, so this
task is the parser, not the contract.

```python
else:
    scan_size_nm = None

pixel_size_nm = scan_size_nm / samps      # TypeError, on the line after the fallback
```

The `else` exists **specifically** to handle a header with no `Scan Size:` field, and then the
next line divides by `samps` unconditionally. The fallback has never worked; mypy has reported
the surrounding signature since M1-T04 (`nanoscope_spm.py:93`), because the function is annotated
`-> np.ndarray` and returns a three-tuple.

---

## Scope

**In scope**

1. `pixel_size_nm` is `None` when `scan_size_nm` is — no division, no crash
2. The return annotation stops lying: `tuple[float | None, float | None, np.ndarray]`
3. The **divisor** is validated, because it is the same line: `Samps/line: 0` is a malformed
   header, not a scan, and today it is a `ZeroDivisionError` from the same expression
4. A header that *states* a non-positive scan size is malformed too — the rule ADR-0025 set for
   the npy loader, applied to the other loader so the two agree
5. **ADR-0026**

**Out of scope**

- The parser's shape. It still takes a path and opens the file twice; the module docstring
  explains why that is left alone until the `ImageLoader` port (M2-T08's successor)
- `lines`. An empty array from `Number of lines: 0` is a degenerate-input question, not a
  division; the characterization already owns degenerate inputs
- Any change to how `Scan Size` is *matched*. The regex, the unit table and the `~m` spelling
  are untouched — this task changes what happens after the match fails, not the match

---

## The decision

Three ways to treat a header with no scan size:

| | |
|---|---|
| **Raise** — a scan without a scale is unusable | Contradicts ADR-0019 and ADR-0025 twice over, and throws away a height map that is perfectly good in pixel space |
| **Default the scale** — 1 nm/px, or the sample count | The defect ADR-0025 deleted from the npy loader, re-entered through the SPM one |
| **Return `None` for both** ✅ | The state now has a defined meaning end to end: `AFMRawData` carries it, `build_substrate_map` accepts it, both detectors report absent nanometres |

A *stated* scan size of `0`, however, is not the same thing as an absent one, and neither is
`Samps/line: 0`. Those are malformed headers and they raise, naming the field — the distinction
ADR-0025 drew between "unknown" and "wrong".

---

## Definition of done

- [ ] A header with no `Scan Size:` returns `(None, None, z)` and the array still loads
- [ ] `Samps/line: 0` and a non-positive stated scan size raise, each naming its field
- [ ] The annotation matches the return; mypy **15 → 14**
- [ ] Tests, including the flip of `test_spm_without_scan_size_crashes_on_the_fallback_it_just_took`
- [ ] `make check` green; delta quantified (expect **zero** — the golden has no SPM phantom)
- [ ] ADR-0026; `STATE.md`, `Progress.md`, `TASKS.md`, `PROJECT_CONTEXT.md`, ADR index
- [ ] Commit: `M3-T17: a header without a scan size parses`

---

## Notes

`test_spm_without_scan_size_crashes_on_the_fallback_it_just_took` pins the defect and says its
assertion flips here. The golden cannot see this module at all — `afm_io` has no phantom — so the
unit tests are the whole of the evidence, which is why they are written first.
