import type { RunView } from "./runs";

/**
 * Where each finished invoice went: what validation found, who settled it,
 * and how it ended. Drives the routing flow and the settled-over-time chart.
 */

export type Validation = "clean" | "warnings" | "errors";
export type Decider = "triage" | "approver" | "vp";
export type Outcome = "paid" | "rejected" | "vp";

export interface Route {
  id: string;
  label: string;
  validation: Validation;
  decider: Decider;
  outcome: Outcome;
}

export const VALIDATION_LABEL: Record<Validation, string> = {
  clean: "All checks passed",
  warnings: "Warnings",
  errors: "Errors",
};
export const DECIDER_LABEL: Record<Decider, string> = {
  triage: "Triage alone",
  approver: "Approver",
  vp: "A VP",
};
export const OUTCOME_LABEL: Record<Outcome, string> = {
  paid: "Paid",
  rejected: "Rejected",
  vp: "Waiting for a VP",
};

/** The route a run took, or null while it is still running (or failed). */
export function routeOf(run: RunView): Route | null {
  const s = run.summary;
  if (!s) return null;
  const outcome: Outcome | null =
    run.status === "paid" ? "paid" : run.status === "rejected" ? "rejected" : run.status === "awaiting_vp" ? "vp" : null;
  if (!outcome) return null;
  const validation: Validation = s.findings.some((f) => f.severity === "error")
    ? "errors"
    : s.findings.some((f) => f.severity === "warning")
      ? "warnings"
      : "clean";
  const traces = run.events.flatMap((e) => (e.type === "trace" ? [e] : []));
  const byVp = run.events.some((e) => e.type === "decision");
  const byTriage = traces.some((e) => e.node === "triage" && /approver was skipped/.test(e.message));
  const decider: Decider = byVp ? "vp" : byTriage ? "triage" : "approver";
  return { id: run.id, label: s.invoice?.invoice_number ?? run.source, validation, decider, outcome };
}

export interface FlowPath {
  key: string;
  steps: [Validation, Decider, Outcome];
  routes: Route[];
}

/** Routes grouped into the distinct paths they took, in a stable order. */
export function flowPaths(routes: Route[]): FlowPath[] {
  const paths = new Map<string, FlowPath>();
  for (const r of routes) {
    const key = `${r.validation}|${r.decider}|${r.outcome}`;
    const p = paths.get(key) ?? { key, steps: [r.validation, r.decider, r.outcome], routes: [] };
    p.routes.push(r);
    paths.set(key, p);
  }
  return [...paths.values()];
}

export interface Settled {
  route: Route;
  // Seconds since the session's first run started.
  at: number;
}

/** When each finished run settled, relative to the first run's start. */
export function settledTimeline(runs: RunView[]): Settled[] {
  const time = (iso?: string) => (iso ? Date.parse(iso) : NaN);
  const starts = runs.flatMap((r) => r.events.filter((e) => e.type === "started").map((e) => time(e.at)));
  const origin = Math.min(...starts.filter((t) => !Number.isNaN(t)));
  if (!Number.isFinite(origin)) return [];
  const out: Settled[] = [];
  for (const run of runs) {
    const route = routeOf(run);
    if (!route) continue;
    const end = [...run.events].reverse().find((e) => e.type === "done" || e.type === "awaiting_vp");
    const t = time(end?.at);
    if (!Number.isNaN(t)) out.push({ route, at: (t - origin) / 1000 });
  }
  return out.sort((a, b) => a.at - b.at);
}
