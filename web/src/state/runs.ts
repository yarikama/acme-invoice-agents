import type { RunEvent, RunOverview, Status, Summary } from "../api/types";

export interface RunView {
  id: string;
  source: string;
  status: Status;
  createdAt: string;
  summary: Summary | null;
  events: RunEvent[];
  error: string | null;
}

export interface RunsState {
  runs: Record<string, RunView>;
  // Newest first.
  order: string[];
  lastSeq: number;
  // The API process the events came from (see RunEvent.boot).
  boot?: string;
}

export type RunsAction = { type: "snapshot"; runs: RunOverview[] } | { type: "event"; event: RunEvent };

export const initialRuns: RunsState = { runs: {}, order: [], lastSeq: -1 };

function blank(id: string, source: string, status: Status, createdAt: string): RunView {
  return { id, source, status, createdAt, summary: null, events: [], error: null };
}

export function runsReducer(state: RunsState, action: RunsAction): RunsState {
  switch (action.type) {
    case "snapshot": {
      // Adds runs the stream hasn't mentioned yet (queued ones); never
      // overwrites what events already told us.
      const runs = { ...state.runs };
      const order = [...state.order];
      for (const r of action.runs) {
        if (!runs[r.id]) {
          runs[r.id] = blank(r.id, r.source, r.status, r.created_at);
          order.push(r.id);
        }
      }
      return { ...state, runs, order: newestFirst(order, runs) };
    }

    case "event": {
      const e = action.event;
      // The API restarted: its log starts again at 0 and holds only what this
      // process knows, so start over from it.
      if (e.boot && state.boot && e.boot !== state.boot) state = { ...initialRuns };
      // The stream replays history on reconnect; skip what we already have.
      if (e.seq <= state.lastSeq) return state;
      if (e.type === "reset") return { ...initialRuns, lastSeq: e.seq, boot: e.boot ?? state.boot };
      const existing = state.runs[e.run_id];
      const run = existing ?? blank(e.run_id, e.source, e.status, e.created_at);
      const next: RunView = {
        ...run,
        status: e.status,
        events: [...run.events, e],
      };
      if (e.type === "done" || e.type === "awaiting_vp") next.summary = e.summary;
      if (e.type === "run_error") next.error = e.message;
      if (e.type === "started" && run.events.length > 0) next.error = null;
      const runs = { ...state.runs, [e.run_id]: next };
      return {
        runs,
        order: existing ? state.order : newestFirst([...state.order, e.run_id], runs),
        lastSeq: e.seq,
        boot: e.boot ?? state.boot,
      };
    }
  }
}

// The snapshot and the stream can arrive in either order; sorting by the
// server's created_at keeps the list stable whichever comes first.
function newestFirst(order: string[], runs: Record<string, RunView>): string[] {
  return [...order].sort((a, b) => runs[b]!.createdAt.localeCompare(runs[a]!.createdAt));
}

export const STATUS_LABEL: Record<Status, string> = {
  queued: "Queued",
  running: "Running",
  awaiting_vp: "Awaiting VP",
  paid: "Paid",
  rejected: "Rejected",
  failed: "Failed",
};
