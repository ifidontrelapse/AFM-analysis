import type { PipelineConfig, Modality, DetectorName, PipelineMode } from "../types/pipeline";

interface Props {
  config: PipelineConfig;
  onChange: (c: PipelineConfig) => void;
  onSubmit: () => void;
  disabled: boolean;
}

function RadioGroup<T extends string>({
  label,
  options,
  value,
  onChange,
  disabledOptions,
}: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  disabledOptions?: Set<T>;
}) {
  return (
    <div className="mb-4">
      <p className="mb-1.5 text-xs font-medium uppercase tracking-widest text-gray-500">{label}</p>
      <div className="flex gap-2 flex-wrap">
        {options.map((o) => {
          const isDisabled = disabledOptions?.has(o.value);
          const isSelected = value === o.value;
          return (
            <button
              key={o.value}
              type="button"
              disabled={isDisabled}
              onClick={() => onChange(o.value)}
              className={`rounded px-3 py-1.5 text-sm transition-colors
                ${isDisabled ? "cursor-not-allowed opacity-30" : "cursor-pointer"}
                ${isSelected && !isDisabled
                  ? "bg-sky-600 text-white"
                  : "bg-gray-800 text-gray-300 hover:bg-gray-700"}`}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-4">
      <div className="flex justify-between mb-1">
        <p className="text-xs font-medium uppercase tracking-widest text-gray-500">{label}</p>
        <span className="text-xs text-gray-400">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-sky-500"
      />
    </div>
  );
}

export default function ConfigPanel({ config, onChange, onSubmit, disabled }: Props) {
  const set = <K extends keyof PipelineConfig>(key: K, val: PipelineConfig[K]) =>
    onChange({ ...config, [key]: val });

  const isAfm = config.modality === "afm";
  const isLog = config.detector === "log";
  const isYolo = config.detector === "yolo";

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-5">
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-widest text-gray-400">
        Configuration
      </h2>

      <RadioGroup<Modality>
        label="Modality"
        value={config.modality}
        options={[
          { value: "afm", label: "AFM" },
          { value: "sem", label: "SEM" },
          { value: "tem", label: "TEM" },
        ]}
        onChange={(v) => {
          const next: PipelineConfig = { ...config, modality: v };
          if (v !== "afm" && next.mode === "baseline") next.mode = "detect";
          onChange(next);
        }}
      />

      <RadioGroup<DetectorName>
        label="Detector"
        value={config.detector}
        options={[
          { value: "log", label: "LoG" },
          { value: "yolo", label: "YOLOv8" },
        ]}
        onChange={(v) => set("detector", v)}
      />

      <RadioGroup<PipelineMode>
        label="Mode"
        value={config.mode}
        options={[
          { value: "detect", label: "Detect" },
          { value: "baseline", label: "Baseline" },
          { value: "segment", label: "Segment" },
        ]}
        disabledOptions={!isAfm ? new Set<PipelineMode>(["baseline"]) : undefined}
        onChange={(v) => set("mode", v)}
      />

      {!isAfm && (
        <div className="mb-4">
          <label className="mb-1.5 block text-xs font-medium uppercase tracking-widest text-gray-500">
            nm / pixel
          </label>
          <input
            type="number"
            min={0}
            step={0.01}
            placeholder="optional"
            value={config.nm_per_pixel ?? ""}
            onChange={(e) =>
              set("nm_per_pixel", e.target.value ? Number(e.target.value) : undefined)
            }
            className="w-full rounded bg-gray-800 px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 outline-none focus:ring-1 focus:ring-sky-600"
          />
        </div>
      )}

      {isLog && (
        <>
          <Slider
            label="LoG percentile"
            min={1}
            max={50}
            step={1}
            value={config.log_percentile ?? 20}
            onChange={(v) => set("log_percentile", v)}
          />
          <Slider
            label="LoG overlap"
            min={0}
            max={1}
            step={0.05}
            value={config.log_overlap ?? 0.3}
            onChange={(v) => set("log_overlap", v)}
          />
        </>
      )}

      {isYolo && (
        <>
          <div className="mb-4 flex items-center justify-between">
            <p className="text-xs font-medium uppercase tracking-widest text-gray-500">YOLO tiling</p>
            <button
              type="button"
              onClick={() => set("yolo_use_tiling", !config.yolo_use_tiling)}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors
                ${config.yolo_use_tiling ? "bg-sky-600" : "bg-gray-700"}`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform
                  ${config.yolo_use_tiling ? "translate-x-4" : "translate-x-0.5"}`}
              />
            </button>
          </div>
          <Slider
            label="YOLO confidence"
            min={0.1}
            max={1}
            step={0.05}
            value={config.yolo_conf ?? 0.5}
            onChange={(v) => set("yolo_conf", v)}
          />
        </>
      )}

      <button
        type="button"
        disabled={disabled}
        onClick={onSubmit}
        className="mt-2 w-full rounded-lg bg-sky-600 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Run Analysis
      </button>
    </div>
  );
}
