import type { PipelineResult } from "../types/pipeline";

interface Props {
  result: PipelineResult | null;
}

export default function ResultViewer({ result }: Props) {
  if (!result) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border-2 border-dashed border-gray-700 bg-gray-900 text-gray-600 text-sm">
        Results will appear here
      </div>
    );
  }

  const hasImage = result.masks_preview_b64.length > 0;

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <div className="relative">
        {hasImage ? (
          <img
            src={`data:image/png;base64,${result.masks_preview_b64}`}
            alt="analysis result"
            className="w-full rounded object-contain"
          />
        ) : (
          <div className="flex h-64 items-center justify-center rounded bg-gray-800 text-sm text-gray-500">
            No preview available
          </div>
        )}

        <span className="absolute right-2 top-2 rounded bg-sky-700/80 px-2 py-0.5 text-xs font-semibold text-white backdrop-blur-sm">
          {result.particle_count} particles
        </span>
      </div>

      {result.pixel_size_nm !== null && (
        <div className="mt-3 flex items-center gap-2 text-xs text-gray-500">
          <div className="h-1 w-16 rounded-full bg-gray-400" />
          <span>{(result.pixel_size_nm * 100).toFixed(1)} nm</span>
        </div>
      )}
    </div>
  );
}
