You are building a Jupyter notebook for automated height measurement of gold 
nanoparticles in Bruker/Veeco Nanoscope AFM images (.001 file extension).

## Project context

AFM .001 files are binary with ASCII header (Nanoscope format). Header contains:
- Data offset, data length, bytes/pixel
- Z scale in format: V [Sens. Zscan] (X.XXX V/LSB) * XX.X nm/V
- Scan size in nm, samples/line, number of lines

The Z-map is a 2D float32 array of height values in nm. Gold nanoparticles appear 
as bright rounded bumps on a flat substrate (mica or silicon).

## Deliverable

Create a single Jupyter notebook: `afm_gold_nanoparticles.ipynb`

The notebook must be:
- Self-contained and linear — run top to bottom with no errors
- Interactive — key parameters in clearly marked CONFIG cells
- Educational — each section has a markdown cell explaining the physics/math behind it
- Robust — every external library import has a try/except with a clear install message

---

## Notebook structure

### Cell 0 — Title markdown
Clean title, one-line description, author/date placeholder, table of contents 
with anchor links to each section.

### Cell 1 — Installation (code, commented out)
All pip install commands commented out, grouped by purpose:
```python
# Core
# !pip install numpy scipy scikit-image matplotlib pandas tqdm

# AFM file reading (try in order)
# !pip install pySPM
# !pip install git+https://github.com/jmarini/nanoscope

# SAM 2 segmentation (optional)
# !pip install git+https://github.com/facebookresearch/sam2.git
# Download checkpoint:
# !wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

### Cell 2 — Imports (code)
All imports in one place. SAM 2 import wrapped in try/except, sets 
SAM_AVAILABLE = True/False. Print confirmation with library versions.

### Cell 3 — ⚙️ CONFIGURATION (code)
Single CFG dict — visually separated, clearly commented:
```python
CFG = {
    # ── FILE ──────────────────────────────
    'file_path':   'your_file.001',   # path to Nanoscope .001 file
    'use_synthetic': True,            # True = generate test data, no real file needed

    # ── PARTICLE SIZE (nm) ────────────────
    'particle_min_nm': 3,
    'particle_max_nm': 80,

    # ── DETECTION SENSITIVITY ─────────────
    'log_threshold': 0.05,            # lower = more particles found (0.01–0.2)
    'log_overlap':   0.3,

    # ── SUBSTRATE ESTIMATION ──────────────
    'opening_radius_px': 15,          # must be > largest particle radius in pixels

    # ── HEIGHT MEASUREMENT ────────────────
    'min_height_nm':      0.5,        # discard anything below this
    'ring_outer_px':      7,          # annular ring width for local baseline
    'ring_inner_erode_px': 2,

    # ── SAM 2 ─────────────────────────────
    'sam_checkpoint': 'sam2.1_hiera_base_plus.pt',
    'sam_config':     'sam2.1_hiera_b+.yaml',
    'device':         'cuda',         # 'cuda' or 'cpu'
    'colormap':       'afmhot',       # try: 'afmhot', 'viridis', 'gray'
    'clip_percentile': 99,
}
```

### Cell 4 — Markdown: "Step 1 — Loading Nanoscope .001 files"
Explain the file format: ASCII header + binary body. Show the key header fields 
visually. Explain z_scale formula: (V/LSB) × (nm/V) = nm/LSB.

### Cell 5 — Loader code
Complete `parse_nanoscope_header(path)` and `read_nanoscope_channel(path)`:
- Parse ASCII header with regex for z_scale, data_offset, samps, lines, bytes_per_pixel, scan_size
- Read int16/int32 binary, reshape, multiply by z_scale → nm
- Compute pixel_size_nm = scan_size_nm / n_cols
- Return z_map (float32), pixel_size_nm, metadata dict
- Three fallback chain: own parser → jmarini/nanoscope → pySPM
- Print found channels with their parameters

`make_synthetic_afm()` for testing:
- 40 gaussian-shaped particles, random heights 2–25 nm, random radii
- Add plane tilt + gaussian noise
- Return z_map, pixel_size_nm=1.0

Execution cell loads data, prints metadata, shows raw AFM image with colorbar.

### Cell 6 — Markdown: "Step 2 — Preprocessing"
Explain plane tilt in AFM, line artifacts from scanner drift. 
Show before/after concept with ASCII art.

### Cell 7 — Preprocessing code
`flatten_plane(z)`: LSQ plane subtraction  
`flatten_lines(z, poly_order=1)`: row-by-row polynomial  
Execute both, show 2-panel figure: before vs after, with Z range in title.

### Cell 8 — Markdown: "Step 3 — Substrate map"
Explain morphological opening: disk SE larger than particles slides under them, 
leaves only substrate. Mention the key parameter: opening_radius_px.

### Cell 9 — Substrate code
`get_substrate_map(z, radius_px)`: morphological opening  
Execute, show 3-panel figure: flattened Z | substrate | z_above = z - substrate  
Add warning if opening_radius_px looks too small relative to image size.

### Cell 10 — Markdown: "Step 4 — Particle detection (LoG)"  
Explain LoG blob detection: scale-space, sigma↔size relationship.  
Formula: radius_nm = sigma_px × √2 × pixel_size_nm

### Cell 11 — Detection code
`detect_seeds(z_above, cfg)`: blob_log on normalized z_above  
Execute, overlay circles on z_above image (cyan circles, radius = sigma×√2).  
Print: N candidates found, sigma range, equivalent diameter range in nm.

### Cell 12 — Markdown: "Step 5 — Height measurement & local baseline"
Explain the core measurement:  
  height = max(Z inside mask) − median(Z in annular ring around mask)  
Show ASCII diagram of the ring, explain why local baseline > global median.

### Cell 13 — Measurement functions code
`get_ring_mask(mask, outer_px, inner_erode_px)` → boolean ring  
`measure_height(z, mask, ...)` → dict with height_nm, mean_nm, baseline_nm, area_px  
`create_circular_mask(shape, cy, cx, r)` → boolean disk  

### Cell 14 — Markdown: "Step 6 — Baseline algorithm (circular masks)"
Explain: use sigma from LoG as radius, create circle, measure height.  
This is the comparison benchmark — simple but fast.

### Cell 15 — Baseline execution
Loop over blobs, create circular mask, measure height, filter by min_height_nm.  
Show progress with tqdm. Build df_baseline DataFrame.  
Print summary stats table.

### Cell 16 — Markdown: "Step 7 — SAM 2 segmentation"
Explain: LoG says WHERE, SAM draws exact WHAT.  
Explain point prompt → mask decoder flow.  
Explain afm_to_rgb conversion and why colormap choice matters.

### Cell 17 — SAM 2 code
`afm_to_rgb(z, colormap, clip_percentile)` → uint8 RGB  
`SAMSegmentor` class with encode(image) and segment(x, y) methods  
Execution cell: if SAM_AVAILABLE → encode image, loop over blobs, segment, filter, measure  
if not SAM_AVAILABLE → print friendly skip message, df_sam = empty DataFrame  

### Cell 18 — Markdown: "Step 8 — Results & comparison"

### Cell 19 — Visualization code
Figure 1: AFM overview — 3 panels (raw | preprocessed | above-substrate)  
Figure 2: Detection — baseline circles overlay on z_above  
Figure 3: SAM masks — colored mask overlay on RGB image (if SAM available)  
Figure 4: Statistics — 2 panels:
  - Height histogram: baseline (blue) vs SAM (gold) overlaid
  - Scatter plot: baseline height vs SAM height for matched particles (if both available)

All figures saved to output directory as PNG.

### Cell 20 — Statistics & comparison code
`print_stats(df, method_name)`: count, mean±std, median, [min, max]  
If both methods available:
  - Match particles by proximity (cKDTree, threshold 10 px)
  - Compute Δheight = SAM − baseline, mean and std
  - Pearson correlation with p-value
  - Print formatted comparison table

### Cell 21 — Export code
Merge df_baseline and df_sam, add physical coordinates in nm.  
Save to CSV: `afm_results_{timestamp}.csv`  
Print final summary and path to saved files.

### Cell 22 — Markdown: "Parameter tuning guide"
Table format:

| Parameter | Effect | Start value | If too many detections | If missing particles |
|-----------|--------|-------------|----------------------|---------------------|
| log_threshold | sensitivity | 0.05 | increase | decrease |
| opening_radius_px | substrate quality | 15 | — | increase if particles cut off |
| ring_outer_px | baseline accuracy | 7 | — | increase for sparse samples |
| colormap | SAM quality | afmhot | try viridis | try gray |

---

## Code quality requirements

- Type hints on all functions
- Docstrings with: what it does, parameters, returns, units
- No magic numbers — all constants via CFG or named variables
- Each executable cell ends with at least one print or plot confirming success
- Figures: figsize proportional to content, titles include key parameter values, 
  axis labels always in nm, colorbars on all AFM images

## Before writing code

1. List all cells in order with one-line description each
2. Identify where pixel_size_nm flows through the computation
3. Check: does the notebook work end-to-end with USE_SYNTHETIC=True and SAM_AVAILABLE=False?
4. Then write cell by cell

## Test file

Place a real .001 file at: ./data/2025/11 Febraury/si-dbs-au-3.007
The notebook should work with USE_SYNTHETIC=True even without this file.