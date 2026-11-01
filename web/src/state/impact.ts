import type { RunView } from "./runs";

/**
 * What the agents changed against the manual process the brief describes
 * (5-day processing, every invoice handled by staff): how fast an invoice
 * reaches a decision, how many need no person, the money held back before
 * it was paid, and what the models cost.
 */
export interface Impact {
  // Runs that reached a decision: paid, rejected, or handed to a VP.
  decided: number;
  // Median seconds from start to that decision; VP waiting time excluded.
  medianSeconds: number | null;
  // Paid or rejected with no VP involved.
  noPerson: number;
  // Rejected invoices and their USD total (other currencies counted only).
  stopped: number;
  stoppedUsd: number;
  // Mean model cost per decided invoice; null if any step was unpriced.
  costPerInvoice: number | null;
}

const time = (iso?: string) => (iso ? Date.parse(iso) : NaN);
const plain = (text?: string | null) => (text ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");

export function impact(runs: RunView[]): Impact {
  const seconds: number[] = [];
  let noPerson = 0;
  let stopped = 0;
  let stoppedUsd = 0;
  let cost: number | null = 0;
  let decided = 0;
  const held = new Set<string>();

  for (const run of runs) {
    const start = time(run.events.find((e) => e.type === "started")?.at);
    const end = run.events.find((e) => e.type === "done" || e.type === "awaiting_vp");
    if (!end) continue;
    decided += 1;
    const t = time(end.at);
    if (!Number.isNaN(start) && !Number.isNaN(t)) seconds.push((t - start) / 1000);

    const vp = run.events.some((e) => e.type === "awaiting_vp");
    if (!vp && (run.status === "paid" || run.status === "rejected")) noPerson += 1;
    if (run.status === "rejected") {
      const inv = run.summary?.invoice;
      // Each invoice once: the same one as a PDF and a JSON, or sent again as
      // a duplicate, is one amount held back.
      const id = `${plain(inv?.vendor)}|${plain(inv?.invoice_number)}`;
      const key = id === "|" ? run.source : id;
      if (!held.has(key)) {
        held.add(key);
        stopped += 1;
        if (inv?.total != null && inv.currency === "USD" && inv.total > 0) stoppedUsd += inv.total;
      }
    }

    for (const e of run.events) {
      if (e.type !== "trace" || !e.usage) continue;
      cost = cost === null || e.usage.cost_usd === null ? null : cost + e.usage.cost_usd;
    }
  }

  seconds.sort((a, b) => a - b);
  const mid = Math.floor(seconds.length / 2);
  const median =
    seconds.length === 0 ? null : seconds.length % 2 ? seconds[mid]! : (seconds[mid - 1]! + seconds[mid]!) / 2;
  return {
    decided,
    medianSeconds: median,
    noPerson,
    stopped,
    stoppedUsd,
    costPerInvoice: decided === 0 || cost === null ? null : cost / decided,
  };
}

/** "9.4s", "1.2 min": how long the agents took. */
export function duration(seconds: number): string {
  return seconds < 60 ? `${seconds.toFixed(1)}s` : `${(seconds / 60).toFixed(1)} min`;
}
