import { useState } from "react";
import { analyze } from "../api/client";
import type { PipelineConfig, PipelineResult } from "../types/pipeline";
import UploadZone from "../components/UploadZone";
import ConfigPanel from "../components/ConfigPanel";
import ResultViewer from "../components/ResultViewer";
import StatsPanel from "../components/StatsPanel";
import Histogram from "../components/Histogram";
import LoadingOverlay from "./LoadingOverlay";

const DEFAULT_CONFIG: PipelineConfig = {
  modality: "afm",
  detector: "log",
  mode: "segment",
};

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [config, setConfig] = useState<PipelineConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await analyze(file, config);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const radii = result?.detections.map((d) => d.radius_nm) ?? [];
  const heightValues = result
    ? result.measurements.map((m) => m.height_nm).filter((v): v is number => v !== undefined)
    : [];
  const areaValues = result
    ? result.measurements.map((m) => m.area_nm2).filter((v): v is number => v !== undefined)
    : [];
  const isAfm = config.modality === "afm";

  return (
    <>
      {loading && <LoadingOverlay />}

      <div className="mx-auto max-w-6xl space-y-6">
        {/* Top row: upload + config */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-4">
            <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500">
              Image
            </h2>
            <UploadZone onFile={setFile} />
          </div>
          <ConfigPanel
            config={config}
            onChange={setConfig}
            onSubmit={handleSubmit}
            disabled={!file || loading}
          />
        </div>

        {error && (
          <div className="rounded-lg border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <>
            <div className="border-t border-gray-800 pt-4">
              <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-gray-500">
                Results
              </p>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <ResultViewer result={result} />
                <div className="space-y-4">
                  <StatsPanel result={result} />
                  {radii.length > 0 && (
                    <Histogram values={radii} label="Radius, nm" color="steelblue" />
                  )}
                  {isAfm && heightValues.length > 0 && (
                    <Histogram values={heightValues} label="Height, nm" color="#22c55e" />
                  )}
                  {!isAfm && areaValues.length > 0 && (
                    <Histogram values={areaValues} label="Area, nm²" color="#a78bfa" />
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}
