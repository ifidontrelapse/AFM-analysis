# Task: build a React web frontend for the nanoparticle analysis pipeline

> **ARCHIVED — historical specification, not a plan.**
>
> This document specified the React client in `frontend/` and the `POST /analyze` backend
> that was never written. The web client is **parked** and the product is a Qt6 desktop
> application — see [ADR-0007](../ADR/ADR-0007-park-web-client.md) and
> [ADR-0002](../ADR/ADR-0002-qt6-desktop-ui.md).
>
> Kept because it is the only record of the intended HTTP contract, including the fields
> the Python side never produced (`masks_preview_b64`, `particle_count`). Moved here from
> the repository root by task M1-T01; it was previously gitignored and therefore
> unshareable.

## Stack

React + TypeScript + Vite + Tailwind CSS.
No UI component library — custom components only, clean scientific aesthetic.
State management: React built-in (`useState`, `useReducer`) — no Redux.
HTTP: native `fetch` API.

---

## Project structure

```
frontend/
├── index.html
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── package.json
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── api/
    │   └── client.ts          ← typed API calls to FastAPI
    ├── types/
    │   └── pipeline.ts        ← TypeScript types mirroring Python dataclasses
    ├── components/
    │   ├── UploadZone.tsx      ← drag & drop file upload
    │   ├── ConfigPanel.tsx     ← detector / mode / params form
    │   ├── ResultViewer.tsx    ← image with overlays
    │   ├── StatsPanel.tsx      ← particle count, mean radius, mean height
    │   └── Histogram.tsx       ← radius / height distribution
    └── pages/
        ├── AnalyzePage.tsx     ← main page (upload + config + results)
        └── LoadingOverlay.tsx  ← full-screen spinner during analysis
```

---

## TypeScript types (`src/types/pipeline.ts`)

Mirror the Python `PipelineResult` and `Detection` dataclasses exactly:

```typescript
export type Modality = "afm" | "sem" | "tem";
export type DetectorName = "log" | "yolo";
export type PipelineMode = "detect" | "baseline" | "segment";

export interface Detection {
  x_px: number;
  y_px: number;
  radius_px: number;
  radius_nm: number;
  confidence: number;
  bbox: [number, number, number, number];   // x1, y1, x2, y2
}

export interface ParticleMeasurement {
  x_px: number;
  y_px: number;
  // AFM fields
  height_nm?: number;
  baseline_nm?: number;
  peak_nm?: number;
  // SEM/TEM fields
  area_nm2?: number;
  radius_nm?: number;
  circularity?: number;
  aspect_ratio?: number;
}

export interface PipelineResult {
  detections: Detection[];
  masks_preview_b64: string;        // base64 PNG — overlay rendered server-side
  measurements: ParticleMeasurement[];
  pixel_size_nm: number | null;
  detector_name: DetectorName;
  mode: PipelineMode;
  modality: Modality;
  particle_count: number;
}

export interface PipelineConfig {
  modality: Modality;
  detector: DetectorName;
  mode: PipelineMode;
  nm_per_pixel?: number;            // SEM/TEM manual scale
  log_overlap?: number;
  log_percentile?: number;
  yolo_use_tiling?: boolean;
  yolo_conf?: number;
}
```

---

## API client (`src/api/client.ts`)

```typescript
const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function analyze(
  file: File,
  config: PipelineConfig,
): Promise<PipelineResult> {
  const form = new FormData();
  form.append("image", file);
  form.append("config", JSON.stringify(config));

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Server error ${res.status}`);
  }

  return res.json();
}
```

---

## Pages and components

### `AnalyzePage.tsx` — layout

Three-column layout on desktop, stacked on mobile:

```
┌─────────────────────────────────────────────────┐
│  [UploadZone]        [ConfigPanel]              │
│                                                 │
│  ─────────────────── Results ─────────────────  │
│  [ResultViewer]      [StatsPanel]               │
│                      [Histogram: radius]        │
│                      [Histogram: height / area] │
└─────────────────────────────────────────────────┘
```

State:
```typescript
const [file, setFile]       = useState<File | null>(null);
const [config, setConfig]   = useState<PipelineConfig>({
  modality: "afm", detector: "log", mode: "segment"
});
const [result, setResult]   = useState<PipelineResult | null>(null);
const [loading, setLoading] = useState(false);
const [error, setError]     = useState<string | null>(null);
```

On submit: call `analyze(file, config)`, set loading, handle error.

---

### `UploadZone.tsx`

- Drag & drop area with dashed border
- Click to open file picker
- Accepted formats: `.spm`, `.npy`, `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`
- Show filename + file size after selection
- Show thumbnail preview for image formats (JPEG/PNG/TIFF)
- Gray out and show "Unsupported format" for anything else

---

### `ConfigPanel.tsx`

Form fields:

| Field | Type | Options / range |
|-------|------|-----------------|
| Modality | radio buttons | AFM / SEM / TEM |
| Detector | radio buttons | LoG / YOLOv8 |
| Mode | radio buttons | detect / baseline (AFM only) / segment |
| nm/pixel | number input | shown only for SEM/TEM, optional |
| LoG percentile | slider 1–50 | shown only when detector=log, default 20 |
| LoG overlap | slider 0–1 | shown only when detector=log, default 0.3 |
| YOLO tiling | toggle | shown only when detector=yolo |
| YOLO conf | slider 0.1–1.0 | shown only when detector=yolo, default 0.5 |

Rules:
- `mode="baseline"` radio is disabled when `modality` is not `"afm"`
- YOLO fields hidden when `detector="log"`, LoG fields hidden when `detector="yolo"`
- Large "Run Analysis" button at the bottom, disabled when no file selected

---

### `ResultViewer.tsx`

Display the result image with overlays:

- Show `masks_preview_b64` as an `<img>` tag (PNG rendered server-side with masks + circles)
- Below the image: pixel scale bar showing physical size in nm
- Particle count badge in top-right corner of the image
- If `result` is null and no loading: show placeholder with dashed border

---

### `StatsPanel.tsx`

Summary statistics table. Columns depend on modality:

**AFM:**
| Stat | Value |
|------|-------|
| Particles detected | N |
| Mean radius | X.X nm |
| Median radius | X.X nm |
| Mean height | X.X nm (only if mode=segment or baseline) |
| Median height | X.X nm |

**SEM/TEM:**
| Stat | Value |
|------|-------|
| Particles detected | N |
| Mean radius | X.X nm (or px if nm_per_pixel unknown) |
| Mean circularity | X.XX |
| Mean aspect ratio | X.XX |

---

### `Histogram.tsx`

Reusable histogram component. Props:
```typescript
interface HistogramProps {
  values: number[];
  label: string;       // x-axis label e.g. "Radius, nm"
  color?: string;      // bar color, default steelblue
  bins?: number;       // default 20
}
```

Render as SVG — no chart library, draw bars manually.
Show median line (gold dashed) and mean line (tomato dashed) with legend.

Show two instances in results:
1. Radius histogram — always (from `detections[].radius_nm`)
2. Height histogram (AFM) or area histogram (SEM/TEM) — when `measurements` non-empty

---

### `LoadingOverlay.tsx`

Full-screen semi-transparent overlay with spinner and text:
```
Analysing image...
```
Shown while `loading === true`.

---

## Environment

`.env` file at repo root:
```
VITE_API_URL=http://localhost:8000
```

---

## `package.json` dependencies

```json
{
  "dependencies": {
    "react": "^18",
    "react-dom": "^18"
  },
  "devDependencies": {
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "@vitejs/plugin-react": "^4",
    "typescript": "^5",
    "vite": "^5",
    "tailwindcss": "^3",
    "autoprefixer": "^10",
    "postcss": "^8"
  }
}
```

---

## Notes for the backend (FastAPI) team

The frontend expects `POST /analyze` to return `PipelineResult` JSON with an extra field:
- `masks_preview_b64: str` — base64-encoded PNG of the image with mask overlay and detection circles drawn on it (equivalent of `plot_pipeline_result` saved to bytes)

If this field is not yet implemented in the backend, return an empty string `""` and
`ResultViewer` will show a placeholder.

---

## Rules

- All components typed with TypeScript — no `any`
- No external chart or UI component libraries — SVG histograms, custom form elements
- Mobile-responsive layout (Tailwind responsive prefixes)
- Error state: show error message below the form, do not crash
- Empty/null states handled in every component