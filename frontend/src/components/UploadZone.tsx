import { useRef, useState, DragEvent, ChangeEvent } from "react";

const ACCEPTED = [".spm", ".npy", ".jpg", ".jpeg", ".png", ".tif", ".tiff"];
const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".tif", ".tiff"]);

function ext(filename: string): string {
  return filename.slice(filename.lastIndexOf(".")).toLowerCase();
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

interface Props {
  onFile: (file: File | null) => void;
}

export default function UploadZone({ onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [unsupported, setUnsupported] = useState(false);

  function handle(f: File) {
    const e = ext(f.name);
    if (!ACCEPTED.includes(e)) {
      setFile(f);
      setPreview(null);
      setUnsupported(true);
      onFile(null);
      return;
    }
    setUnsupported(false);
    setFile(f);
    onFile(f);

    if (IMAGE_EXTS.has(e)) {
      const url = URL.createObjectURL(f);
      setPreview(url);
    } else {
      setPreview(null);
    }
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handle(f);
  }

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) handle(f);
  }

  return (
    <div
      className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 transition-colors cursor-pointer select-none
        ${dragging ? "border-sky-400 bg-sky-400/10" : "border-gray-600 bg-gray-900 hover:border-gray-400"}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept={ACCEPTED.join(",")}
        onChange={onChange}
      />

      {!file && (
        <>
          <svg className="mb-3 h-10 w-10 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <p className="text-sm text-gray-400">Drag & drop or click to select</p>
          <p className="mt-1 text-xs text-gray-600">{ACCEPTED.join(", ")}</p>
        </>
      )}

      {file && (
        <div className="w-full" onClick={(e) => e.stopPropagation()}>
          {preview && (
            <img
              src={preview}
              alt="preview"
              className="mb-3 max-h-48 w-full rounded object-contain"
            />
          )}
          <div className={`rounded px-3 py-2 text-sm ${unsupported ? "bg-red-900/40 text-red-400" : "bg-gray-800 text-gray-300"}`}>
            <span className="font-medium">{file.name}</span>
            <span className="ml-2 text-xs text-gray-500">{formatBytes(file.size)}</span>
            {unsupported && (
              <p className="mt-1 text-xs text-red-400">Unsupported format</p>
            )}
          </div>
          <button
            className="mt-2 text-xs text-gray-500 underline hover:text-gray-300"
            onClick={() => { setFile(null); setPreview(null); setUnsupported(false); onFile(null); }}
          >
            Remove
          </button>
        </div>
      )}
    </div>
  );
}
