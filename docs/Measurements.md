# Measurements — what every number means

**Status:** active · **Created:** 2026-08-17 (M7-T10, ADR-0079) · **Checked by:**
`tests/unit/test_measurement_docs.py`

This is the reference for the numbers this application produces: what each column is the measurement
*of*, in what units, computed on which array, and where two producers answer the same question
differently. It describes the code as it is. Where the code and this file disagree, the code wins and
this file is wrong — say so in the same pull request (PROJECT_RULES §8).

A test asserts that every column `measurement_columns()` can declare appears here, and that this file
names no column the schema does not have. A column added without a paragraph fails the gate.

---

## 1. Where measurements come from

Two entirely different things in this project are called a measurement, and they are not
interchangeable:

| | An **analysis run** | A **hand tool** |
|---|---|---|
| Produced by | a detector and a measurement pass (`run_pipeline`) | an operator dragging on the canvas |
| Stored as | `results/run_<id>/measurements.csv`, indexed by the database (ADR-0042) | rows in `rulers` (schema v8, ADR-0074) |
| Re-runnable | yes — delete it and run again | **no.** Nobody else drew that line |
| Columns | vary by producer (§3, ADR-0031) | two endpoints and a kind |

§2 to §6 are the analysis run's table. §7 is the hand tools.

---

## 2. Conventions that hold everywhere

- **Arrays are indexed `[y, x]`.** Always. Coordinates in a *table* are `x_px`, `y_px`, and boxes are
  `(x1, y1, x2, y2)` (PROJECT_RULES §3).
- **Suffixes are units:** `_px` pixels, `_nm` nanometres, `_nm2` square nanometres. A name without a
  suffix is dimensionless (`circularity`, `aspect_ratio`) or a label (`method`).
- **An unknown scale is `None`, never `0` and never the pixel value** (ADR-0019, ADR-0025). A scan
  imported without a pixel size still measures: what it loses is every `_nm` column derived from
  pixel size.
- **Heights are calibrated by the *z* axis, not by the pixel size.** A scan with no `pixel_size_nm`
  still reports `height_nm` — the two calibrations are independent, which is why an unscaled scan
  keeps its heights and loses its radii (M6-T06).
- **`.spm` carries its own z calibration; `.npy` carries none.** An `.npy` array is taken to be
  nanometres as it stands, and nothing checks that. If the file holds volts, every `_nm` height in
  the table is volts wearing a nanometre's name.
- **A measurement table is a subset of the detections** (ADR-0033): rows are dropped, so a run with
  40 detections may measure 37. §6 lists every reason.

---

## 3. The table's shape: a core plus blocks

`measurement_columns()` declares the columns and their dtypes in one place, for all four producers
(ADR-0031). Every row of every table carries the **core**; each **block** is present in full or
absent in full, and `method` says which producer wrote the row and therefore which blocks to expect.

| Block | When it is present | Columns |
|---|---|---|
| core | always | `particle_id`, `x_px`, `y_px`, `area_px`, `method` |
| detector | the measurement was prompted by a detection | `sigma_px`, `detector_radius_nm` |
| height | AFM — a height map was available | `height_nm`, `baseline_nm`, `peak_nm`, `mean_nm`, `baseline_source`, `ring_px` |
| geometry | a real mask was measured | `radius_px`, `radius_nm`, `area_nm2`, `circularity`, `aspect_ratio` |
| segmentation | a segmenter scored its own mask | `mask_score` |

There is no wide table with NaN where a producer cannot fill a column: that would say SEM/TEM *has*
heights and they are all missing. It has none — the modality does not produce one.

| `method` | Producer | Blocks |
|---|---|---|
| `baseline_circle` | `measure_all_baseline` — LoG blobs, circular masks, AFM | core + detector + height |
| `sam2_blobs` | SAM2 prompted by LoG blob centres | core + detector + height *or* geometry + segmentation |
| `sam2_boxes` | SAM2 prompted by detector boxes | core + detector + height *or* geometry + segmentation |

---

## 4. The core

**`particle_id`** — an integer identifying the row within its table. **It does not mean the same
thing across producers** (defect **B-069**): `measure_all_baseline` writes the *blob's* index, so it
indexes the detections and has gaps where a row was dropped; the SAM2 producers write the index of
the row being appended, which renumbers after every drop. Do not join two producers' tables on it,
and do not read it as "detection number" unless `method` is `baseline_circle`.

**`x_px`, `y_px`** — the particle's centre in pixels, float64. For `baseline_circle` it is the
**rounded** blob centre, because the circular mask was built at those integer pixels and that is
where the measurement happened; the SAM2 producers write the sub-pixel prompt point.

**`area_px`** — pixels in the mask that was measured. **Which mask depends on `method`, and this is
the biggest trap in the table:**

- `baseline_circle`: the area of a **disk drawn from the detector's sigma**, `π(σ√2)²`. It is a
  function of what the detector estimated, not a measurement of the particle.
- `sam2_*`: the area of the **actual segmentation**.

Comparing the two as if they were one quantity compares a detector's guess against a measurement.

**`method`** — which producer wrote the row. The discriminator for everything above.

---

## 5. The blocks

### 5.1 Detector — what prompted the measurement

**`sigma_px`** — the scale at which the Laplacian-of-Gaussian responded. The circular mask the
baseline producer measures has radius `σ√2` px, which is the standard relation between a LoG's
scale and the blob radius it responds to.

**`detector_radius_nm`** — `σ√2 × pixel_size_nm`: the radius the *detector* thought this particle
had, in nanometres. **NaN throughout when the scale is unknown.** It is deliberately not called
`radius_nm`: one is where we looked, the other is what we found, and calling both `radius_nm` is the
defect ADR-0031 was written to remove (D-17).

### 5.2 Height — AFM only

Every height in this table is **relative**: how far a particle stands above its own local substrate.

```
height_nm = peak − baseline
```

**`peak_nm`** — the maximum z inside the particle's mask, in the height map's own units, *not*
baseline-subtracted. `peak_nm = height_nm + baseline_nm` by construction; it is reported so no
consumer has to reconstruct it.

**`baseline_nm`** — the local substrate level under this particle: the **median** z over a cleaned
ring around the mask. The ring is the mask dilated by `inner_erode_px` (to get off the particle's
slope) then dilated again by `outer_px`, minus the inner part, intersected with the substrate mask —
so *neighbouring particles are removed whether or not the detector found them*. Defaults:
`measure_outer_px=5`, `measure_inner_erode_px=3` for the baseline producer,
`sam2_outer_ring_px=5`, `sam2_inner_erode_px=2` for SAM2.

**`ring_px`** — how many pixels that cleaned ring had. Fewer than `min_ring_px` (default 5) means the
local baseline is not trustworthy, and the two producers then part company (below).

**`baseline_source`** — `"ring"` or `"global"`. **`"global"` means this particle's own surroundings
could not be used** and the median of the whole image's substrate stood in. A row marked `global` in
a dense field is the row to distrust first.

**`mean_nm`** — the mean z inside the mask, minus the same baseline. Sensitive to how much substrate
the mask includes, which is exactly what differs between producers.

**The two producers do not measure the same height.** Same formula, different quantities in it:

| | `baseline_circle` | `sam2_*` |
|---|---|---|
| Peak over | the **circular** mask from the detector's sigma | the **eroded real** mask (`binary_erosion`, `inner_erode_px`) |
| Mean over | the same circular mask | the same eroded mask |
| Ring too small | falls back to the **global** substrate median, `baseline_source="global"` | **skips the particle** — so `baseline_source` is always `"ring"` |

A circular mask over a non-circular particle includes substrate, which lowers `mean_nm` and leaves
`height_nm` (a maximum) largely alone. Compare heights across methods with that in mind; compare
`mean_nm` across methods only if you have a reason.

### 5.3 Geometry — a measured mask's shape

Requires a real segmentation. The circular-mask producer does **not** emit this block: the geometry
of a circle drawn from a radius is that radius, and recording it as a measurement would be circular
in both senses.

**`radius_px`** — the **equivalent-area radius**: the radius of a circle with the same area as the
mask (`equivalent_diameter_area / 2`). Not a fitted radius and not half the longest axis.

**`radius_nm`** — `radius_px × nm_per_pixel`, or `None` when the scale is unknown.

**There is no diameter column.** The project reports radii; a diameter is `2 × radius_px` or
`2 × radius_nm`, and the equivalent-area diameter of a real particle is what that gives you. Where a
paper asks for "particle size", say which of the two you used.

**`area_nm2`** — `area_px × nm_per_pixel²`, or `None` when the scale is unknown.

**`circularity`** — `4π·area / perimeter²`, dimensionless, `1.0` for a mathematically perfect circle.
**Read it comparatively, not absolutely:**

- a digitised disk of radius 10 px scores **0.916**, not 1.0, because a rasterised outline is longer
  than the circle it approximates. Values near 0.9 are round particles, not defective ones;
- a mask so small that its perimeter is 0 gets a substituted perimeter of 1.0, and a **single pixel
  scores 12.57**. Filed as **B-071**.

**`aspect_ratio`** — the region's major axis divided by its minor axis, `≥ 1` by construction, `1.0`
for a circle. **A mask with no minor axis also reports `1.0`** — a one-pixel-wide line, the most
elongated thing there is, is reported as perfectly round. Filed as **B-071**; until it is fixed, treat
`aspect_ratio == 1.0` as *"round, or unmeasurable"*.

### 5.4 Segmentation

**`mask_score`** — the segmenter's own predicted IoU for the mask it produced: *how sure the
segmenter is of this outline*. **Not** the detector's confidence, which lives on the detection
(`Detection.confidence`, ADR-0028). Two scores, two names, on purpose.

---

## 6. What the table leaves out

A run's detections and its measured rows are different counts, for these reasons and no others:

| Dropped | Where | Why |
|---|---|---|
| the circular mask has fewer than 4 pixels | `measure_all_baseline` | the mask ran past the image edge |
| `height_nm` is not `> 0` | `measure_all_baseline` | a particle below its own substrate is an artefact — and `not h > 0` rather than `h <= 0` **because NaN passes `<=`** (B-059, ADR-0033) |
| the cleaned ring has fewer than 5 pixels | SAM2 producers | no trustworthy local baseline, and this path has no global fallback |
| the mask is empty | SAM2 SEM/TEM | nothing was segmented |

A scan where nothing survives returns **an empty table with the full schema and the declared
dtypes** — not a table with no columns (ADR-0027, D-08). "No measurable particles" and "no analysis"
are different statements, and both are said in the same shape.

When the substrate mask comes back empty — an Otsu threshold on a map whose values do not
separate — there is no global baseline, every particle without a usable ring is dropped, and the run
**warns**, because the alternative reads as "there was nothing here" (ADR-0033).

---

## 7. What an operator measures by hand

Neither of these is in `measurements.csv`, and neither is an annotation: a line has no area, and
ADR-0044's shapes are refused without one. They are `rulers` rows, and **the length is never
stored** — it is arithmetic over the two endpoints, and a stored copy is a second answer waiting to
disagree (ADR-0074).

**Distance** (`kind = "distance"`) — `hypot(x₂−x₁, y₂−y₁)` in pixels, and `× pixel_size_nm` for
nanometres. Without a scale the panel says **"scale unknown"** and shows pixels; it does not show a
number that implies a scale nobody recorded. A non-positive scale *raises*, because absent and wrong
are different (ADR-0025). Two clicks in the same place measure 0, which is an answer.

**The scale is the loaded scan's, not the row's** (ADR-0083). They agree for anything imported since
that decision, because the import records what the file states; for a project imported before it,
the row can say *unknown* about a Nanoscope file whose header states a scale — and a ruler is drawn
over the array, so it is the array's scale that converts it.

**Height profile** (`kind = "profile"`) — the heights under the line, **one sample per pixel of
length plus the far end** (a line from x=10 to x=14 crosses five pixels, not four), sampled
**bilinearly**, clamped at the array's edges rather than extrapolated. For a horizontal line with
integer endpoints this is exactly `z[y, x1:x2+1]`, which is what
`notebooks/afm_gold_nanoparticles.ipynb` §5.1 does — the case M7's exit criterion names, asserted as
equality (ADR-0075).

**A profile names the stage it measured.** Raw and flattened maps give different numbers, and a
measurement whose provenance is a checkbox somebody set four clicks ago is not one anybody can
defend (ADR-0061).

---

## 8. Known defects that change what a number means

| ID | What |
|---|---|
| **B-069** | `particle_id` means the blob's index in one producer and the row counter in another |
| **B-071** | `aspect_ratio` and `circularity` substitute constants where the value is undefined — `1.0` for a line with no minor axis, 12.57 for a single pixel |
| **B-072** | the geometry block reads skimage properties deprecated in 0.26 and removed in 2.0 (same values, new names) |
| **B-062** | the LoG detector finds nothing on a low-SNR phantom — a table with no rows is not always "no particles" |

---

## References

- `nanoscope/core/science/measurement/schema.py` — the columns, declared once
- `nanoscope/core/science/measurement/height.py` — the ring, the baseline, the peak
- `nanoscope/core/science/measurement/geometry.py` — the shape metrics
- `nanoscope/core/science/metrology.py` — the hand tools' arithmetic
- ADR-0031 (one measurement schema), ADR-0033 (a NaN height is not a measurement), ADR-0025 (an
  unknown scale is not a fabricated one), ADR-0074 / ADR-0075 (the hand tools), ADR-0079 (this file)
