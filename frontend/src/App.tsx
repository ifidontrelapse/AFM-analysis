import AnalyzePage from "./pages/AnalyzePage";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-lg font-semibold tracking-wide text-gray-200">
          Nanoparticle Analysis
        </h1>
      </header>
      <main className="px-4 py-6 md:px-8">
        <AnalyzePage />
      </main>
    </div>
  );
}
