interface HistogramProps {
  values: number[];
  label: string;
  color?: string;
  bins?: number;
}

function mean(v: number[]): number {
  return v.reduce((a, b) => a + b, 0) / v.length;
}

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
}

export default function Histogram({
  values,
  label,
  color = "steelblue",
  bins = 20,
}: HistogramProps) {
  if (values.length === 0) return null;

  const W = 400;
  const H = 160;
  const PAD = { top: 10, right: 16, bottom: 32, left: 36 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binSize = range / bins;

  const counts = Array<number>(bins).fill(0);
  for (const v of values) {
    const i = Math.min(Math.floor((v - min) / binSize), bins - 1);
    counts[i]++;
  }

  const maxCount = Math.max(...counts);

  const xScale = (v: number) => PAD.left + ((v - min) / range) * chartW;
  const yScale = (c: number) => PAD.top + chartH - (c / maxCount) * chartH;

  const meanVal = mean(values);
  const medianVal = median(values);

  const barW = (chartW / bins) - 1;

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" aria-label={label}>
        {/* bars */}
        {counts.map((c, i) => {
          const x = PAD.left + i * (chartW / bins);
          const y = yScale(c);
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={barW}
              height={chartH - (y - PAD.top)}
              fill={color}
              opacity={0.75}
            />
          );
        })}

        {/* mean line */}
        {meanVal >= min && meanVal <= max && (
          <line
            x1={xScale(meanVal)} y1={PAD.top}
            x2={xScale(meanVal)} y2={PAD.top + chartH}
            stroke="tomato" strokeWidth={1.5} strokeDasharray="4 3"
          />
        )}

        {/* median line */}
        {medianVal >= min && medianVal <= max && (
          <line
            x1={xScale(medianVal)} y1={PAD.top}
            x2={xScale(medianVal)} y2={PAD.top + chartH}
            stroke="gold" strokeWidth={1.5} strokeDasharray="4 3"
          />
        )}

        {/* x-axis */}
        <line
          x1={PAD.left} y1={PAD.top + chartH}
          x2={PAD.left + chartW} y2={PAD.top + chartH}
          stroke="#4b5563" strokeWidth={1}
        />

        {/* x-axis ticks */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = min + t * range;
          const x = PAD.left + t * chartW;
          return (
            <g key={t}>
              <line x1={x} y1={PAD.top + chartH} x2={x} y2={PAD.top + chartH + 4} stroke="#6b7280" />
              <text x={x} y={PAD.top + chartH + 14} textAnchor="middle" fontSize={9} fill="#9ca3af">
                {v.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* x-axis label */}
        <text
          x={PAD.left + chartW / 2}
          y={H - 2}
          textAnchor="middle"
          fontSize={10}
          fill="#6b7280"
        >
          {label}
        </text>

        {/* legend */}
        <g transform={`translate(${PAD.left + chartW - 90}, ${PAD.top + 2})`}>
          <line x1={0} y1={5} x2={14} y2={5} stroke="tomato" strokeWidth={1.5} strokeDasharray="4 3" />
          <text x={17} y={9} fontSize={8} fill="#d1d5db">mean</text>
          <line x1={0} y1={17} x2={14} y2={17} stroke="gold" strokeWidth={1.5} strokeDasharray="4 3" />
          <text x={17} y={21} fontSize={8} fill="#d1d5db">median</text>
        </g>
      </svg>
    </div>
  );
}
