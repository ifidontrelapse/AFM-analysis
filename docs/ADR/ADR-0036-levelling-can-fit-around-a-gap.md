# ADR-0036 — Levelling can fit around a gap

- **Status:** Accepted
- **Date:** 2026-08-08
- **Affects:** `nanoscope/core/science/preprocessing/flatten.py`,
  `nanoscope/core/validation.py` · **B-060** · M3-T25
- **Numerical impact:** **5 golden differences, all of them the new `gapped_levelling` block.**
  Nothing recorded moves: `allow_gaps` defaults to `False`, so every existing call is
  byte-identical.

## Context

M3-T13 made a non-finite value a rejection — *"a height map must be finite"* — enforced by
`ensure_height_map` at fourteen entry points. That was the honest reading of what the code already
did: `flatten_plane` had always refused NaN, through `scipy.lstsq`'s own message, while
`flatten_lines` propagated it and `detect_particles` answered "no particles". Making the contract
uniform was right. ADR-0030 also said, in its own text, that rejecting was *not the best behaviour
available*, and filed the alternative as **B-060**.

**A dropped scan line is a real artefact, not malformed input.** An AFM that loses feedback for
two lines produces a map with two rows of NaN and four thousand good ones, and today the whole
scan is refused.

### What a masked fit recovers — measured

A 64×64 tilted scan with four 8 nm particles and two dropped rows, levelled three ways and
compared against **levelling the same scan with no gap at all**:

| Strategy | plane coefficients `(a, b, c)` | max error on the intact pixels |
|---|---|---:|
| ungapped reference | `0.051149  0.031149  0.712928` | — |
| **masked fit** | `0.051186  0.031076  0.739255` | **0.0287 nm** |
| `nan_to_num(z, 0.0)` | `0.049587  0.031377  0.677039` | 0.1343 nm |

The masked fit recovers the plane to 0.03 nm. Zero-filling is **4.7× worse**, and the interesting
part is *why*: look at the tilt coefficient, `0.0496` against the true `0.0511`. Filling with
zeros does not add noise — it tells the fit that the sample dips to zero along two lines, and the
plane leans to accommodate it.

## Decision

**Levelling accepts gaps when the caller asks: `allow_gaps=False` by default.**

```python
flatten_plane(z, *, allow_gaps=False)
flatten_lines(z, poly_order=1, *, allow_gaps=False)
ensure_height_map(z, name="z", *, allow_gaps=False)
```

**Opt-in, not automatic.** Accepting NaN silently would put the library back where D-15 found it:
levelling tolerating what detection refuses, two functions disagreeing about what an image is.
That disagreement is the defect ADR-0030 closed six tasks ago, and it is not worth re-opening for
an artefact the caller knows about in advance. The default path does not move a single number.

**The gap stays absent in the output.** NaN in, NaN out, in exactly the same pixels. Not filled,
not interpolated: an interpolated value is a measurement nobody made, and this milestone has
deleted seven substitute values (ADR-0019, 0024, 0025, 0027, 0028, 0031, 0032). Interpolation is a
*feature*, with its own decision about the method — filed as **B-066**.

**A row that cannot be fitted comes back absent, and is counted out loud.** A fully-NaN row is
exactly what a dropped scan line is; a row with fewer than `poly_order + 1` finite points has no
fit either. Both return NaN, and the number of such rows is warned about — rows vanishing without
a reason is how B-059 stayed invisible for a milestone.

**A row that is too *sparse* is a gap; a row that is too *short* is still an error.** The first is
a fact about the data, the second is a malformed request, and `InvalidParameterError` still covers
the second exactly as M3-T13 left it.

## Consequences

**Positive**

- A scan with dropped lines can be levelled instead of discarded, and the result says where the
  data is missing.
- The comparison that justifies it is in the golden, not only in this file: the masked fit, the
  ungapped reference and the zero-filled alternative are all recorded per phantom.
- `ensure_height_map` gained a parameter rather than a hole: the contract still has one default
  and one place that states it.

**Negative**

- **This does not make the pipeline gap-tolerant**, and that is the honest headline. The levelled
  output still carries NaN, so `build_substrate_map` and both detectors still refuse it. What the
  caller gains is a levelled map they can crop, inspect or fill deliberately, instead of an
  exception. **Filed as B-065**, and pinned by a test so the limitation cannot be forgotten.
- A second code path inside each levelling function. The mitigation is a test asserting the two
  are identical on an intact map — if `allow_gaps` ever becomes a different implementation of
  levelling rather than the same one, that test fails.

**Neutral**

- No phantom has a gap, so nothing recorded moves. The file grows by the probes.

## What is deliberately not in this commit

- **Gap-tolerant substrate, detection and measurement — B-065.** It needs a decision about what a
  substrate *under* a gap is: the morphological opening will happily propagate NaN across the
  structuring element, so the answer is not "pass NaN through" but a real choice, and it is science
  rather than plumbing.
- **Interpolation — B-066.** Nearest-neighbour, linear along the fast axis and inpainting give
  three different answers, and choosing needs the evaluation harness plus a view on what an
  operator expects to see.
- **Detecting gaps in the loader.** `_read_nanoscope_z` produces no NaN today; a scan with dropped
  lines arrives as whatever the instrument wrote, which is a separate investigation.

## The measured delta

**5 differences, one per AFM phantom, all `gapped_levelling: ADDED`.** Nothing moves — the
default path is untouched by construction, and that was the prediction the plan made before the
measurement.

What the block records, with two dropped rows cut into each phantom:

| Phantom | masked fit error | zero-filled error | ratio | rows absent | default path |
|---|---:|---:|---:|---:|---|
| `afm_flat_monodisperse` | 0.0037 nm | 0.0056 nm | 1.5× | 2 | `InvalidImageError` |
| `afm_tilted_polydisperse` | 0.0471 nm | **0.1983 nm** | **4.2×** | 2 | `InvalidImageError` |
| `afm_dense_overlapping` | 0.0144 nm | 0.0244 nm | 1.7× | 2 | `InvalidImageError` |
| `afm_sparse_low_snr` | 0.0009 nm | 0.0012 nm | 1.3× | 2 | `InvalidImageError` |
| `afm_coarse_pixels` | 0.0797 nm | 0.0924 nm | 1.2× | 2 | `InvalidImageError` |

**The advantage tracks the tilt, and that confirms the mechanism rather than merely restating
it.** `afm_tilted_polydisperse` is the phantom built with a slope, and it is the one where
zero-filling costs 4.2×; on the flat phantoms the two strategies are within 20–50 % of each other.
That is exactly what the coefficient table above predicts — the fill corrupts the *plane*, so its
damage is proportional to how much plane there is to get wrong. On a scan with no tilt there is
little to corrupt; on a real AFM scan, which is tilted by construction, there is.

The absolute errors are small everywhere (0.0009–0.08 nm) because the phantoms are 256 px and
two rows are 0.8 % of them. The synthetic scene in the tests, at 64 px, loses 3 % of its rows and
shows 0.029 nm against 0.134 nm — the same story with a bigger gap.

`flatten_plane_default_on_a_gapped_map` records `InvalidImageError` on all five, which is
ADR-0030's contract holding. If that ever becomes something else, it reads as drift rather than
as an improvement.

## Alternatives considered

| Alternative | Why not | Would be acceptable if |
|---|---|---|
| Accept NaN automatically in levelling | Re-opens D-15's disagreement — levelling tolerates what detection refuses — for a case the caller knows about in advance | The whole pipeline handled gaps (B-065) |
| Fill with zeros and level normally | Measured: 4.7× the error, and it biases the tilt rather than adding noise, because zero is a value and the fit believes it | The gap value were physically meaningful |
| Fill with the row median, then level | Better than zero and still an invented measurement, and it hides the gap from every downstream reader | The output marked the filled pixels |
| Drop the gapped rows and level the rest | Changes the array's shape, so every coordinate downstream shifts — a worse defect than the one being fixed | The rows were at the edge |
| Relax `ensure_height_map` globally | Undoes ADR-0030 at fourteen sites to serve one artefact at two | The finiteness rule had been wrong rather than incomplete |

## Compliance

- `tests/unit/test_gapped_levelling.py` — **12 tests** in four classes. *The default is
  unchanged*: a gapped map is still refused with ADR-0030's message, and an intact map levels
  **byte-identically with and without the flag** — the guard against `allow_gaps` quietly becoming
  a second implementation of levelling. *The plane fits around the gap*: it recovers the ungapped
  answer to 0.05 nm, beats zero-filling by more than 3×, leaves exactly the input's pixels absent,
  and refuses a map with fewer than three finite pixels. *The rows fit around the gap*: a
  partially gapped row levels from what is left, a fully gapped one comes back absent, the count
  is warned about, an intact scan says nothing, and a row too *sparse* for its order is a gap
  rather than an error. *What this does not do*: the levelled output is still refused by
  `build_substrate_map`, asserted as a test so the limitation cannot be forgotten.
- Golden: `gapped_levelling` on all five AFM phantoms, carrying the masked fit, the ungapped
  reference and the zero-filled comparison — so the file holds the evidence the decision was made
  on, not only its outcome.

## References

- `ADR-0030` — the finiteness contract, unchanged as the default; this adds a named exception the
  caller must ask for, which is the difference between an escape hatch and a hole
- `ADR-0029` — the promotion rule `flatten_lines` follows, unaffected
- **B-060**, closed here; **B-065**, **B-066**, filed here
