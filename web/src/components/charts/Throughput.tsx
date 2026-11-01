import { useState } from "react";

import { spreadLabels } from "../../lib/labels";
import { type Outcome, OUTCOME_LABEL, type Settled } from "../../state/routing";
import { useWidth } from "../../state/useWidth";
import { Tip } from "./Tip";

const SERIES: Outcome[] = ["paid", "vp", "rejected"];
const L = 34;
const R = 150;
const T = 10;
const B = 28;
// One line of 15px label text, so stacked end labels never touch.
const LINE = 18;

function clock(s: number): string {
  if (s < 10) return `${s.toFixed(1).replace(/\.0$/, "")}s`;
  return s < 60 ? `${Math.round(s)}s` : `${(s / 60).toFixed(1).replace(/\.0$/, "")} min`;
}

/**
 * Invoices settled over the session, one step line per outcome, labelled at
 * its end. Hover for the counts at any moment.
 */
export function Throughput({ settled }: { settled: Settled[] }) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [at, setAt] = useState<{ t: number; x: number; y: number } | null>(null);
  const W = Math.max(width, 360);
  const H = 240;
  const end = Math.max(settled.at(-1)?.at ?? 1, 1);
  const counts = SERIES.map((o) => settled.filter((s) => s.route.outcome === o).length);
  const maxY = Math.max(2, ...counts);
  const x = (t: number) => L + (t / end) * (W - L - R);
  const y = (v: number) => H - B - (v / maxY) * (H - T - B);
  const step = maxY <= 6 ? 1 : maxY <= 12 ? 2 : 5;
  const ticksY = Array.from({ length: Math.floor(maxY / step) + 1 }, (_, i) => i * step);
  // The smallest step that keeps the time axis to six ticks or fewer.
  const tickEvery = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((st) => end / st <= 6) ?? 1200;
  const ticksX = Array.from({ length: Math.floor(end / tickEvery) + 1 }, (_, i) => i * tickEvery);

  const line = (o: Outcome) => {
    let c = 0;
    let d = `M${x(0)},${y(0)}`;
    for (const s of settled) if (s.route.outcome === o) d += ` H${x(s.at)} V${y(++c)}`;
    return `${d} H${x(end)}`;
  };
  const doneBy = (t: number) => settled.filter((s) => s.at <= t);
  // Lines often end on the same count (two at 0): stack their labels.
  const labelY = spreadLabels(counts.map(y), LINE, T + 6, H - B);

  return (
    <div ref={ref} className="relative">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Invoices settled over time">
        {ticksY.map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="chart-grid" />
            <text x={L - 8} y={y(v) + 5} textAnchor="end" className="chart-axis">
              {v}
            </text>
          </g>
        ))}
        {ticksX.map((t) => (
          <text key={t} x={x(t)} y={H - 6} textAnchor="middle" className="chart-axis">
            {t === 0 ? "0" : clock(t)}
          </text>
        ))}
        {SERIES.map((o, i) => (
          <g key={o}>
            <path d={line(o)} className={`flow-line flow-${o}`} />
            {Math.abs(labelY[i]! - y(counts[i]!)) > 1 && (
              // A moved label keeps a short leader back to its line's end.
              <path
                d={`M${x(end) + 5},${y(counts[i]!)} L${x(end) + 9},${labelY[i]!}`}
                className={`flow-line flow-leader flow-${o}`}
              />
            )}
            <circle cx={x(end)} cy={y(counts[i]!)} r={4} className={`flow-dot flow-${o}`} />
            <text x={x(end) + 12} y={labelY[i]! + 5} className="chart-label">
              {OUTCOME_LABEL[o]} {counts[i]}
            </text>
          </g>
        ))}
        {at && <line x1={x(at.t)} x2={x(at.t)} y1={T} y2={H - B} className="chart-cross" />}
        <rect
          x={L}
          y={T}
          width={W - L - R}
          height={H - T - B}
          fill="transparent"
          onMouseMove={(e) => {
            const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
            const px = e.clientX - box.left;
            const t = Math.max(0, Math.min(end, ((px - L) / (W - L - R)) * end));
            setAt({ t, x: px, y: e.clientY - box.top });
          }}
          onMouseLeave={() => setAt(null)}
        />
      </svg>
      {at && (
        <Tip x={Math.min(at.x + 12, W - 220)} y={at.y + 12}>
          <b>
            {clock(at.t)}: {doneBy(at.t).length} of {settled.length} settled
          </b>
          {SERIES.map((o) => (
            <div key={o}>
              {OUTCOME_LABEL[o]}: {doneBy(at.t).filter((s) => s.route.outcome === o).length}
            </div>
          ))}
        </Tip>
      )}
    </div>
  );
}
