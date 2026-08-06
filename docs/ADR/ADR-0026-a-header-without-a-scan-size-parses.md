# ADR-0026 — A header without a scan size parses

- **Status:** Accepted
- **Date:** 2026-08-06
- **Affects:** `nanoscope/core/science/io/nanoscope_spm.py` · **M3-T17** (the third face of audit
  **D-07**)
- **Numerical impact:** **none — 0 golden differences.** The golden has no SPM phantom; this
  module's evidence is its unit tests, which is itself worth recording.

## Context

The SPM parser has always had a branch for a header with no `Scan Size:` field, and that branch
has never worked:

```python
else:
    scan_size_nm = None

pixel_size_nm = scan_size_nm / samps      # TypeError, one line later
```

The fallback crashes on the branch it has just taken. mypy has been reporting the same function
since M1-T04 — annotated `-> np.ndarray`, returning a three-tuple — which is the static shadow of
a function nobody had read closely.

Two things make this the right moment rather than an arbitrary one. First, **the state now has a
meaning**: ADR-0019 made `None` survivable in the detectors, ADR-0025 made it honest in the npy
loader and gave it a defined behaviour through `AFMRawData`, `PreprocessingResult` and
`build_substrate_map`. Second, **the same expression has a second failure mode**: `samps` is the
divisor, and a header stating `Samps/line: 0` produces a `ZeroDivisionError` from the same line —
an error that names nothing.

## Decision

**A header with no `Scan Size` yields `(None, None, z)`.** No division, no substitute, no crash.
The height map is decoded and returned exactly as it always was; only the metadata is absent.

```python
pixel_size_nm = None if scan_size_nm is None else scan_size_nm / samps
```

**A header that *states* an impossible size is rejected, and says which field.** `Scan Size: 0`
and `Samps/line: 0` are malformed headers, not unknown scales:

```python
raise ValueError(f"header states a non-positive Samps/line: {samps}")
raise ValueError(f"header states a non-positive Scan Size: {scan_size_nm} nm")
```

This is the distinction ADR-0025 drew at the npy loader — `None` is a state, zero is a caller
error — applied to the other loader so that the two agree about what a scale is. A file is not a
caller, but the principle transfers exactly: absent and wrong are different, and only one of them
is recoverable.

**The annotation stops lying.** `-> tuple[float | None, float | None, np.ndarray]`, with a
docstring stating that the first two are `None` *together*. mypy: **15 → 14**.

## Consequences

**Positive**

- **D-07 is closed on all three faces.** Detectors (M3-T11), the npy loader (M3-T20), and now the
  SPM header. Every route to "no scale" ends in the same state rather than in three different
  exceptions.
- A real scan whose header lost its `Scan Size` — which is what this branch was written for —
  now yields a usable height map instead of a `TypeError`.
- Two silent crashes become two errors that name their field, per PROJECT_RULES §3.
- One fewer mypy error, and it was this defect's.

**Negative**

- **A file with a malformed `Scan Size: 0` now fails where it used to produce `0.0` nm/px.** That
  is intended — every physical value derived from it was zero — but it is a behaviour change for
  any file in `data/` carrying such a header. None of the 628 scans read for ADR-0024 did.
- The parser grows two guards, in a function the module docstring already describes as legacy
  shape awaiting the `ImageLoader` port. They are cheap and they are on the line the task is
  about, but they are not a redesign.

**Neutral**

- **Zero golden difference, and none was possible**: `afm_io` has no phantom, because the
  characterization set is synthetic arrays rather than synthetic files. The 28 unit tests in
  `tests/unit/test_afm_io.py` are the entire safety net for this module, which is worth stating
  plainly rather than leaving as an absence.
- The `Scan Size` regex, the unit table and the `~m` spelling are untouched. This changes what
  happens *after* a failed match, not the match.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Raise when the header has no scan size | Contradicts ADR-0019 and ADR-0025, and throws away a height map that is entirely usable in pixel space. It also makes the `else` branch pointless — it exists to tolerate this header | AFM output were only ever consumed in nanometres |
| Default to 1 nm/px, or to `samps` | The defect ADR-0025 has just deleted from the npy loader, re-entering through the SPM one. A fabricated scale is indistinguishable from a measured one | — |
| Return `scan_size_nm=None` but keep computing `pixel_size_nm` from something else | There is nothing else. The pixel size *is* the scan size divided by the sample count; with the numerator absent the quotient is absent | The header carried an independent pixel-size field |
| Treat `Samps/line: 0` and `Scan Size: 0` as "unknown" as well | Then a corrupt header is indistinguishable from an incomplete one, and the loader keeps two spellings for `None` | Zero were a physically meaningful value for either field |
| Leave the zero guards to M3-T13's error taxonomy | They are on the exact line this task fixes, and M3-T13 is about typed exceptions across every entry point, not about which values are legal here. The taxonomy can retype these two later | The guards needed a new exception class to exist |

## Compliance

- `tests/unit/test_afm_io.py` — the characterization test flips from *crashes on the fallback it
  just took* to *reports an unknown scale*, and asserts the array still decodes to `(LINES,
  SAMPS)` `float32`; two new tests reject `Samps/line: 0` and `Scan Size: 0 0 nm` by message.
  **Restoring the division turns 3 red.**
- Golden: unchanged, 0 differences, and the reason is recorded above.

## References

- `docs/audit/2026-07-28-baseline-audit.md` §D-07
- `ADR-0019` — unknown scale is a state, not a crash (the detectors)
- `ADR-0025` — the npy loader, and the "absent versus wrong" distinction this ADR reuses
- **M1-T04** — where mypy first reported this function's signature
