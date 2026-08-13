# ADR-0075 — A profile is the notebook's slice, extended with a stated rule

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** operator + agent (M7-T06)
- **Affects:** `core/science/metrology`, `application/use_cases/metrology`, `gui/panels` · M7

## Context

M7's third exit criterion names its own reference:

> *"A height profile along a drawn line matches the notebook implementation."*

So the first job was to read the notebook. `notebooks/afm_gold_nanoparticles.ipynb` §5.1 does exactly
this:

```python
profile_crop = z_flat[y_i, x_i-half : x_i+half]
coords       = (np.arange(2*half) - half) * pixel_size_nm
```

**A horizontal row slice, no interpolation**, plotted against distance in nanometres.

## Decision

### 1. The criterion covers one case, and that case is asserted as equality

For an axis-aligned line with integer endpoints, `height_profile` returns **exactly** the pixels the
notebook slices — `np.array_equal`, not `allclose`. That is what "matches the notebook" can honestly
mean, and it is what the test says.

### 2. An arbitrary line is an extension, and the rule is stated

The notebook has no diagonal case, so there is nothing to match: this is new behaviour with a
decision behind it. **Bilinear**, because a diagonal profile made of nearest-neighbour steps is a
picture of the sampling rather than of the sample — and a test asserts the one thing nearest-neighbour
cannot do: a line half-way between a row of 0 and a row of 10 reads 5 all the way along.

### 3. One sample per pixel of length, plus the far end

Which is what makes §1 an equality rather than an approximation: a line from x=10 to x=14 crosses
five pixels, not four.

### 4. The sampling clamps at the edges rather than extrapolating

A line whose end is half a pixel outside the scan is an operator's aim; inventing values beyond the
data would be a measurement of nothing.

### 5. It validates its input like every other numerical entry point

`ensure_height_map` at the door — ADR-0030's funnel at its fifteenth site. A profile of a 3-D array
or of a NaN map is a wrong answer waiting to be plotted.

### 6. The profile names the stage it measured

Profiling a raw map and a flattened one give different numbers, and both are legitimate questions.
ADR-0061 made the stage visible in the viewer; a profile that did not name it would be a measurement
whose provenance is a checkbox somebody set four clicks ago.

## Consequences

**Positive** — the exit criterion is met in the only way it can be honestly met, with an equality
where the reference exists and a stated rule where it does not; the plot carries its own provenance;
`ruler.kind` from M7-T05 has its second reader, as designed.

**Negative** — the profile is recomputed on every redraw, including on a stage change. At a few
hundred samples that is nothing; a scan-length line on a 4096² map is 4096 bilinear samples per
repaint, and the first thing to cache if it ever shows.

**Neutral** — no roughness, no step height, no fitted line. Each is a scientific claim, and each
would need an ADR and a test against something better than an eye.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Nearest-neighbour sampling | A diagonal profile of steps is a picture of the sampling |
| Match the notebook exactly, horizontal lines only | An operator draws where the feature is, not where the axis is |
| Extrapolate beyond the scan | A measurement of data that does not exist |
| Profile the raw file always | Denies the flattened question, which is the more common one |
| Profile without naming the stage | A number whose provenance is a checkbox four clicks ago |

## Compliance

`tests/unit/test_metrology.py` asserts equality with the notebook's row slice (and its column
equivalent), the distance axis, the bilinear rule through a case nearest-neighbour cannot pass,
clamping at the edge, and the two refusals. `tests/gui/test_profile_panel.py` asserts the plot reads
the scan under the line, **names the stage**, falls back to pixels without a scale, and follows the
selection.

## References

- `notebooks/afm_gold_nanoparticles.ipynb` §5.1 — the reference the criterion names
- ADR-0074 — the ruler this reads a second way
- ADR-0061 — the stage this names
- ADR-0030 — the validation funnel, fifteenth site
- ADR-0025 — pixels when there is no scale
