import type { Status } from "../api/types";

export type Filter = "all" | "awaiting_vp" | "active" | "paid" | "rejected" | "failed";

export const SEGMENTS: { key: Filter; label: string; statuses: Status[] }[] = [
  { key: "all", label: "All", statuses: [] },
  { key: "awaiting_vp", label: "Awaiting VP", statuses: ["awaiting_vp"] },
  { key: "active", label: "In progress", statuses: ["queued", "running"] },
  { key: "paid", label: "Paid", statuses: ["paid"] },
  { key: "rejected", label: "Rejected", statuses: ["rejected"] },
  { key: "failed", label: "Failed", statuses: ["failed"] },
];

export function matches(filter: Filter, status: Status): boolean {
  const seg = SEGMENTS.find((s) => s.key === filter);
  return !seg || seg.statuses.length === 0 || seg.statuses.includes(status);
}
