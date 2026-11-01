import type { ReactNode } from "react";

/** A hover tooltip, placed inside the chart's relative box. */
export function Tip({ x, y, children }: { x: number; y: number; children: ReactNode }) {
  return (
    <div className="chart-tip" style={{ left: x, top: y }} role="status">
      {children}
    </div>
  );
}
