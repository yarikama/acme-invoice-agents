import type { Status } from "../api/types";

export type Filter = "all" | "awaiting_vp" | "active" | "paid" | "rejected" | "failed";

export const SEGMENTS: { key: Filter; label: string; light: string; statuses: Status[] }[] = [
  { key: "all", label: "All", light: "lt-queued", statuses: [] },
  { key: "awaiting_vp", label: "Awaiting VP", light: "lt-awaiting_vp", statuses: ["awaiting_vp"] },
  { key: "active", label: "In progress", light: "lt-running", statuses: ["queued", "running"] },
  { key: "paid", label: "Paid", light: "lt-paid", statuses: ["paid"] },
  { key: "rejected", label: "Rejected", light: "lt-rejected", statuses: ["rejected"] },
  { key: "failed", label: "Failed", light: "lt-failed", statuses: ["failed"] },
];

export function matches(filter: Filter, status: Status): boolean {
  const seg = SEGMENTS.find((s) => s.key === filter);
  return !seg || seg.statuses.length === 0 || seg.statuses.includes(status);
}
