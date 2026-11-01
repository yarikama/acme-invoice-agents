import { describe, expect, it } from "vitest";

import type { RunEvent, Summary } from "../api/types";
import { duration, impact } from "./impact";
import type { RunView } from "./runs";

const at = (s: number) => new Date(Date.UTC(2026, 0, 1, 0, 0, s)).toISOString();

function run(id: string, status: RunView["status"], events: Partial<RunEvent>[], total = 100): RunView {
  const summary = { invoice: { total, currency: "USD" } } as unknown as Summary;
  return {
    id,
    source: id,
    status,
    createdAt: at(0),
    summary,
    error: null,
    events: events.map((e, i) => ({
      id: i,
      seq: i,
      run_id: id,
      source: id,
      created_at: at(0),
      status,
      ...e,
    })) as RunEvent[],
  };
}

const cost = (usd: number | null) => ({
  type: "trace" as const,
  usage: { input_tokens: 10, output_tokens: 5, cost_usd: usd },
});

describe("impact", () => {
  it("measures the runs against the manual process", () => {
    const got = impact([
      run("paid", "paid", [{ type: "started", at: at(0) }, cost(0.01), { type: "done", at: at(4) }]),
      run("bad", "rejected", [{ type: "started", at: at(0) }, cost(0.03), { type: "done", at: at(10) }], 5000),
      // Waited for a VP: decided when it paused, and a person was involved.
      run(
        "vp",
        "paid",
        [
          { type: "started", at: at(0) },
          { type: "awaiting_vp", at: at(6) },
          { type: "done", at: at(600) },
        ],
        900,
      ),
      run("busy", "running", [{ type: "started", at: at(0) }]),
    ]);
    expect(got).toEqual({
      decided: 3,
      medianSeconds: 6,
      noPerson: 2,
      stopped: 1,
      stoppedUsd: 5000,
      costPerInvoice: 0.04 / 3,
    });
  });

  it("leaves the cost unknown when a step was unpriced, and is empty with no runs", () => {
    const got = impact([run("a", "paid", [{ type: "started", at: at(0) }, cost(null), { type: "done", at: at(2) }])]);
    expect(got.costPerInvoice).toBeNull();
    expect(impact([])).toMatchObject({ decided: 0, medianSeconds: null, costPerInvoice: null });
  });

  it("formats durations", () => {
    expect(duration(9.43)).toBe("9.4s");
    expect(duration(90)).toBe("1.5 min");
  });
});
