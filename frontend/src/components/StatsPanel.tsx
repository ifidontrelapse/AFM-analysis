import type { PipelineResult } from "../types/pipeline";

interface Props {
  result: PipelineResult;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function fmt(v: number, decimals = 1): string {
  return v.toFixed(decimals);
}

interface Row {
  label: string;
  value: string;
}

export default function StatsPanel({ result }: Props) {
  const radii = result.detections.map((d) => d.radius_nm);
  const hasNm = result.pixel_size_nm !== null;

  const rows: Row[] = [];

  rows.push({ label: "Particles detected", value: String(result.particle_count) });

  if (result.modality === "afm") {
    rows.push({ label: "Mean radius", value: `${fmt(mean(radii))} nm` });
    rows.push({ label: "Median radius", value: `${fmt(median(radii))} nm` });

    const heights = result.measurements
      .map((m) => m.height_nm)
      .filter((v): v is number => v !== undefined);

    if (heights.length > 0 && (result.mode === "segment" || result.mode === "baseline")) {
      rows.push({ label: "Mean height", value: `${fmt(mean(heights))} nm` });
      rows.push({ label: "Median height", value: `${fmt(median(heights))} nm` });
    }
  } else {
    const radiusLabel = hasNm ? "nm" : "px";
    const radiusValues = hasNm
      ? radii
      : result.detections.map((d) => d.radius_px);

    rows.push({ label: "Mean radius", value: `${fmt(mean(radiusValues))} ${radiusLabel}` });

    const circularities = result.measurements
      .map((m) => m.circularity)
      .filter((v): v is number => v !== undefined);
    if (circularities.length > 0) {
      rows.push({ label: "Mean circularity", value: fmt(mean(circularities), 2) });
    }

    const aspects = result.measurements
      .map((m) => m.aspect_ratio)
      .filter((v): v is number => v !== undefined);
    if (aspects.length > 0) {
      rows.push({ label: "Mean aspect ratio", value: fmt(mean(aspects), 2) });
    }
  }

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-widest text-gray-500">
        Statistics
      </h3>
      <table className="w-full text-sm">
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-b border-gray-800 last:border-0">
              <td className="py-1.5 text-gray-400">{r.label}</td>
              <td className="py-1.5 text-right font-mono text-gray-200">{r.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
