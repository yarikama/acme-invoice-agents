import { useState } from "react";

import {
  DECIDER_LABEL,
  type FlowPath,
  OUTCOME_LABEL,
  type Route,
  VALIDATION_LABEL,
} from "../../state/routing";
import { useWidth } from "../../state/useWidth";
import { Tip } from "./Tip";

const COLUMNS = [
  { title: "Validation", keys: ["clean", "warnings", "errors"], label: VALIDATION_LABEL as Record<string, string> },
  { title: "Decided by", keys: ["triage", "approver", "vp"], label: DECIDER_LABEL as Record<string, string> },
  { title: "Outcome", keys: ["paid", "vp", "rejected"], label: OUTCOME_LABEL as Record<string, string> },
] as const;

const NODE_W = 8;
const NODE_GAP = 16;
const LABEL_W = 150;
const TOP = 30;

interface Node {
  x: number;
  y: number;
  h: number;
  count: number;
  label: string;
  idx: number;
}

/**
 * Where invoices went, as an alluvial flow: one band per distinct path, from
 * what validation found, to who settled it, to how it ended. Bands are
 * coloured by outcome; hover one to follow it.
 */
export function RoutingFlow({ paths, total }: { paths: FlowPath[]; total: number }) {
  const { ref, width } = useWidth<HTMLDivElement>();
  const [hot, setHot] = useState<{ key: string; x: number; y: number } | null>(null);
  const W = Math.max(width, 420);
  const H = Math.max(220, Math.min(420, 90 + total * 16));
  const usable = H - TOP - 8;
  const colX = COLUMNS.map((_, i) => i * ((W - LABEL_W - NODE_W) / (COLUMNS.length - 1)));

  // Nodes: only those some path passes through, sized by count.
  const nodes: Record<string, Node> = {};
  COLUMNS.forEach((col, ci) => {
    const present = col.keys.filter((k) => paths.some((p) => p.steps[ci] === k));
    const counts = present.map((k) => paths.filter((p) => p.steps[ci] === k).reduce((s, p) => s + p.routes.length, 0));
    const unit = (usable - NODE_GAP * Math.max(present.length - 1, 0)) / Math.max(total, 1);
    let y = TOP;
    present.forEach((k, i) => {
      nodes[`${ci}:${k}`] = { x: colX[ci]!, y, h: counts[i]! * unit, count: counts[i]!, label: col.label[k]!, idx: i };
      y += counts[i]! * unit + NODE_GAP;
    });
  });
  const unit = (usable - NODE_GAP * 2) / Math.max(total, 1);

  // Stack each node's bands by the whole path, so bands cross only where routes do.
  const rank = (p: FlowPath) => p.steps.map((k, ci) => nodes[`${ci}:${k}`]!.idx);
  const sorted = [...paths].sort((a, b) => {
    const A = rank(a), B = rank(b);
    for (let i = 0; i < A.length; i++) if (A[i] !== B[i]) return A[i]! - B[i]!;
    return 0;
  });
  const slot: Record<string, number> = {};
  COLUMNS.forEach((_, ci) => {
    const cursor: Record<string, number> = {};
    for (const p of sorted) {
      const k = p.steps[ci]!;
      const node = nodes[`${ci}:${k}`]!;
      const y = cursor[k] ?? node.y;
      slot[`${p.key}#${ci}`] = y;
      cursor[k] = y + p.routes.length * (node.h / node.count);
    }
  });

  const describe = (p: FlowPath) =>
    `${p.routes.length} ${p.routes.length === 1 ? "invoice" : "invoices"}: ${VALIDATION_LABEL[p.steps[0]]} → ${DECIDER_LABEL[p.steps[1]]} → ${OUTCOME_LABEL[p.steps[2]]}`;
  const hotPath = sorted.find((p) => p.key === hot?.key);

  return (
    <div ref={ref} className="relative">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="How invoices were routed" className={hot ? "flow-dim" : undefined}>
        {COLUMNS.map((c, i) => (
          <text key={c.title} x={colX[i]} y={14} className="chart-stage">
            {c.title}
          </text>
        ))}
        {sorted.map((p) => {
          const t = p.routes.length * unit;
          const gap = Math.min(1, t / 4);
          let d = "";
          for (let ci = 0; ci < COLUMNS.length - 1; ci++) {
            const n0 = nodes[`${ci}:${p.steps[ci]}`]!, n1 = nodes[`${ci + 1}:${p.steps[ci + 1]}`]!;
            const h0 = p.routes.length * (n0.h / n0.count), h1 = p.routes.length * (n1.h / n1.count);
            const x0 = colX[ci]! + NODE_W, x1 = colX[ci + 1]!, mx = (x0 + x1) / 2;
            const y0 = slot[`${p.key}#${ci}`]! + gap, y1 = slot[`${p.key}#${ci + 1}`]! + gap;
            d += `M${x0},${y0} C${mx},${y0} ${mx},${y1} ${x1},${y1} L${x1},${y1 + h1 - gap * 2} C${mx},${y1 + h1 - gap * 2} ${mx},${y0 + h0 - gap * 2} ${x0},${y0 + h0 - gap * 2} Z `;
          }
          return (
            <path
              key={p.key}
              d={d}
              className={`flow-band flow-${p.steps[2]} ${hot?.key === p.key ? "flow-hot" : ""}`}
              tabIndex={0}
              aria-label={describe(p)}
              onMouseMove={(e) => {
                const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                setHot({ key: p.key, x: e.clientX - box.left, y: e.clientY - box.top });
              }}
              onMouseLeave={() => setHot(null)}
              onFocus={() => setHot({ key: p.key, x: W / 2, y: TOP })}
              onBlur={() => setHot(null)}
            />
          );
        })}
        {Object.entries(nodes).map(([key, n]) => (
          <g key={key}>
            <rect x={n.x} y={n.y} width={NODE_W} height={Math.max(n.h, 2)} rx={4} className="flow-node" />
            <text x={n.x + NODE_W + 8} y={n.y + n.h / 2 - 2} className="chart-label">
              {n.label}
            </text>
            <text x={n.x + NODE_W + 8} y={n.y + n.h / 2 + 16} className="chart-count">
              {n.count}
            </text>
          </g>
        ))}
      </svg>
      {hot && hotPath && (
        <Tip x={Math.min(hot.x + 12, W - 260)} y={hot.y + 12}>
          <b>{describe(hotPath)}</b>
          <br />
          <span className="mono">{hotPath.routes.map((r: Route) => r.label).join(", ")}</span>
        </Tip>
      )}
    </div>
  );
}
