import type { PipelineConfig, PipelineResult } from "../types/pipeline";

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
    throw new Error((err as { detail?: string }).detail ?? `Server error ${res.status}`);
  }

  return res.json() as Promise<PipelineResult>;
}
