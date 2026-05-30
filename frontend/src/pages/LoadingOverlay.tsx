export default function LoadingOverlay() {
  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gray-950/80 backdrop-blur-sm">
      <div className="h-10 w-10 animate-spin rounded-full border-4 border-gray-700 border-t-sky-500" />
      <p className="mt-4 text-sm text-gray-400 tracking-wide">Analysing image...</p>
    </div>
  );
}
