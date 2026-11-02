import { describe, expect, it } from "vitest";

import type { RunEvent, Summary } from "../api/types";
import { pipeline, soFar, traceLines } from "./pipeline";
import { initialRuns, runsReducer, type RunsState } from "./runs";

let seq = 0;
function ev(partial: Partial<RunEvent> & { type: RunEvent["type"] }, runId = "r1"): RunEvent {
  const s = seq++;
  return {
    id: s,
    seq: s,
    run_id: runId,
    source: "invoice_1014.xml",
    created_at: `2026-01-01T00:00:0${runId === "r1" ? 1 : 2}Z`,
    status: "running",
    ...partial,
  } as RunEvent;
}

const summary = (over: Partial<Summary> = {}): Summary => ({
  status: "paid",
  source_format: "xml",
  invoice: null,
  findings: [],
  route: "auto",
  draft: null,
  payment: null,
  reason: null,
  trace: [],
  ...over,
});

function play(events: RunEvent[], state: RunsState = initialRuns): RunsState {
  return events.reduce((s, event) => runsReducer(s, { type: "event", event }), state);
}

describe("runsReducer", () => {
  it("builds a run from its events alone", () => {
    seq = 0;
    const state = play([
      ev({ type: "started" }),
      ev({ type: "trace", node: "ingest", stage: "ingest", message: "Parsed XML in code." }),
      ev({ type: "done", status: "paid", summary: summary() }),
    ]);
    const run = state.runs.r1!;
    expect(run.status).toBe("paid");
    expect(run.events).toHaveLength(3);
    expect(run.summary?.status).toBe("paid");
    expect(state.order).toEqual(["r1"]);
  });

  it("ignores events replayed after a reconnect", () => {
    seq = 0;
    const first = [ev({ type: "started" }), ev({ type: "progress", stage: "ingest", step: "start" })];
    const once = play(first);
    const twice = play(first, once);
    expect(twice).toBe(once);
  });

  it("adds queued runs from a snapshot without clobbering streamed ones", () => {
    seq = 0;
    const streamed = play([ev({ type: "started" })]);
    const state = runsReducer(streamed, {
      type: "snapshot",
      runs: [
        {
          id: "r1",
          source: "x",
          status: "queued",
          created_at: "2026-01-01T00:00:00Z",
          invoice_number: null,
          vendor: null,
          total: null,
          currency: null,
        },
        {
          id: "r2",
          source: "y",
          status: "queued",
          created_at: "2030-01-01T00:00:00Z",
          invoice_number: null,
          vendor: null,
          total: null,
          currency: null,
        },
      ],
    });
    expect(state.runs.r1!.status).toBe("running");
    expect(state.runs.r2!.status).toBe("queued");
  });

  it("orders runs by the server's created_at, whatever arrives first", () => {
    seq = 0;
    const state = play([ev({ type: "started" }, "r1"), ev({ type: "started" }, "r2")]);
    expect(state.order).toEqual(["r2", "r1"]);
    const snap = runsReducer(state, {
      type: "snapshot",
      runs: [
        {
          id: "r0",
          source: "z",
          status: "queued",
          created_at: "2025-01-01T00:00:00Z",
          invoice_number: null,
          vendor: null,
          total: null,
          currency: null,
        },
      ],
    });
    expect(snap.order).toEqual(["r2", "r1", "r0"]);
  });

  it("clears an old error when a run starts again", () => {
    seq = 0;
    const state = play([ev({ type: "run_error", status: "failed", message: "402" }), ev({ type: "started" })]);
    expect(state.runs.r1!.error).toBeNull();
  });
});

describe("pipeline", () => {
  it("shows the running segment and nothing past it", () => {
    seq = 0;
    const state = play([
      ev({ type: "started" }),
      ev({ type: "trace", node: "ingest", stage: "ingest", message: "Parsed XML in code." }),
      ev({ type: "progress", stage: "validate", step: "start" }),
    ]);
    const states = Object.fromEntries(pipeline(state.runs.r1!).map((s) => [s.key, s.state]));
    expect(states).toMatchObject({ ingest: "done", validate: "run", draft: "off", pay: "off" });
  });

  it("waits at the VP and flags warnings", () => {
    seq = 0;
    const paused = summary({
      status: "awaiting_vp",
      findings: [{ code: "FOREIGN_CURRENCY", severity: "warning", title: "Billed in EUR", detail: "" }],
    });
    const state = play([
      ev({ type: "trace", node: "ingest", stage: "ingest", message: "Parsed XML in code." }),
      ev({ type: "trace", node: "validate", stage: "validate", message: "0 errors, 1 warning." }),
      ev({ type: "trace", node: "review", stage: "approve", message: "Draft 1: escalate." }),
      ev({ type: "trace", node: "critique", stage: "approve", message: "Critic agrees." }),
      ev({ type: "awaiting_vp", status: "awaiting_vp", request: {}, summary: paused }),
    ]);
    const states = Object.fromEntries(pipeline(state.runs.r1!).map((s) => [s.key, s.state]));
    expect(states).toMatchObject({ validate: "flag", critique: "done", vp: "wait", pay: "off" });
  });

  it("marks where a failed run stopped, and skips the VP when it was not needed", () => {
    seq = 0;
    const failed = play([
      ev({ type: "progress", stage: "ingest", step: "start" }),
      ev({ type: "run_error", status: "failed", message: "needs an LLM" }),
    ]);
    expect(pipeline(failed.runs.r1!)[0]!.state).toBe("fail");

    seq = 0;
    const paid = play([
      ev({ type: "trace", node: "ingest", stage: "ingest", message: "ok" }),
      ev({ type: "trace", node: "pay", stage: "pay", message: "mock_payment returned success." }),
      ev({ type: "done", status: "paid", summary: summary() }),
    ]);
    const states = Object.fromEntries(pipeline(paid.runs.r1!).map((s) => [s.key, s.state]));
    expect(states).toMatchObject({ vp: "skip", pay: "done" });
  });

  it("shows a triage-settled run as drafted, with no critique", () => {
    seq = 0;
    const state = play([
      ev({ type: "trace", node: "validate", stage: "validate", message: "0 errors." }),
      ev({ type: "trace", node: "triage", stage: "triage", message: "jev says approve." }),
      ev({ type: "trace", node: "pay", stage: "pay", message: "mock_payment returned success." }),
      ev({ type: "done", status: "paid", summary: summary() }),
    ]);
    const states = Object.fromEntries(pipeline(state.runs.r1!).map((s) => [s.key, s.state]));
    expect(states).toMatchObject({ draft: "done", critique: "skip", pay: "done" });
    // The decision model is a model even when the approver runs offline.
    expect(traceLines(state.runs.r1!, false)[1]!.voice).toBe("model");
  });

  it("calls a validation stop Blocked, not Failed", () => {
    seq = 0;
    const rejected = summary({
      status: "rejected",
      findings: [{ code: "UNKNOWN_ITEM", severity: "error", title: "Item not in inventory", detail: "" }],
    });
    const state = play([
      ev({ type: "trace", node: "ingest", stage: "ingest", message: "ok" }),
      ev({ type: "trace", node: "validate", stage: "validate", message: "1 error, 0 warnings." }),
      ev({ type: "trace", node: "reject", stage: "pay", message: "Not paid." }),
      ev({ type: "done", status: "rejected", summary: rejected }),
    ]);
    const validate = pipeline(state.runs.r1!).find((s) => s.key === "validate")!;
    expect(validate).toMatchObject({ state: "block", word: "Blocked" });
  });
});

describe("traceLines", () => {
  it("tells model, code and human apart", () => {
    seq = 0;
    const state = play([
      ev({ type: "trace", node: "ingest", stage: "ingest", message: "Attempt 1: extracted with the LLM." }),
      ev({ type: "trace", node: "validate", stage: "validate", message: "0 errors." }),
      ev({ type: "trace", node: "review", stage: "approve", message: "Draft 1: approve." }),
      ev({ type: "decision", approved: true, note: "ok" }),
      ev({ type: "trace", node: "human_review", stage: "approve", message: "VP approved. Note: ok" }),
    ]);
    const voices = traceLines(state.runs.r1!, true).map((l) => l.voice);
    expect(voices).toEqual(["model", "code", "model", "human"]);
    expect(traceLines(state.runs.r1!, false)[2]!.voice).toBe("code");
  });

  it("shows tool calls and guardrail send-backs as code lines", () => {
    seq = 0;
    const state = play([
      ev({
        type: "trace",
        node: "review",
        stage: "approve",
        tool: "check_inventory",
        message: "check_inventory(GadgetX): GadgetX: 5 in stock, list price $750.00.",
      }),
      ev({
        type: "trace",
        node: "review",
        stage: "approve",
        tool: "guardrail",
        message: "Sent the decision back. Missing: OUT_OF_STOCK.",
      }),
      ev({ type: "trace", node: "review", stage: "approve", message: "Draft 1: reject." }),
    ]);
    const lines = traceLines(state.runs.r1!, true);
    expect(lines.map((l) => [l.node, l.voice])).toEqual([
      ["tool", "code"],
      ["guardrail", "code"],
      ["review", "model"],
    ]);
    // Tool lines still light the draft segment of the pipeline.
    expect(pipeline(state.runs.r1!).find((s) => s.key === "draft")!.state).toBe("done");
  });
});

describe("soFar", () => {
  const invoice = {
    invoice_number: "INV-1",
    revision: null,
    vendor: "Widgets Inc.",
    vendor_address: null,
    invoice_date: null,
    due_date: null,
    currency: "USD",
    line_items: [{ item: "WidgetA", quantity: 2, unit_price: 5, amount: null, note: null }],
    subtotal: null,
    tax_amount: null,
    shipping: null,
    total: 10,
    payment_terms: null,
    notes: null,
  };
  const finding = { code: "OUT_OF_STOCK", severity: "error" as const, title: "Out of stock", detail: "FakeItem: 0" };

  it("fills in from each step's trace line before the run is done", () => {
    seq = 0;
    const trace = (node: string, data: object) =>
      ev({ type: "trace", node, stage: node, message: "x", data } as Partial<RunEvent> & { type: "trace" });
    const afterIngest = play([ev({ type: "started" }), trace("ingest", { invoice })]).runs.r1!;
    expect(soFar(afterIngest)).toEqual({ invoice, findings: null });
    const afterValidate = play([trace("validate", { findings: [finding] })], {
      ...initialRuns,
      runs: { r1: afterIngest },
      order: ["r1"],
    }).runs.r1!;
    expect(soFar(afterValidate)).toEqual({ invoice, findings: [finding] });
    // The line also carries the step's output into its conversation turn.
    expect(traceLines(afterValidate, true).map((l) => l.output)).toEqual([{ invoice }, { findings: [finding] }]);
  });

  it("prefers the run's summary once it has one", () => {
    seq = 0;
    const done = play([ev({ type: "done", status: "paid", summary: summary({ invoice, findings: [] }) })]).runs.r1!;
    expect(soFar(done)).toEqual({ invoice, findings: [] });
  });
});

describe("reset", () => {
  it("clears every run when someone starts over, and ignores older events", () => {
    seq = 0;
    const before = play([ev({ type: "started" }), ev({ type: "done", status: "paid", summary: summary() })]);
    expect(before.order).toEqual(["r1"]);
    const reset = ev({ type: "reset" } as Partial<RunEvent> & { type: "reset" });
    const after = play([reset], before);
    expect(after.order).toEqual([]);
    expect(after.lastSeq).toBe(reset.seq);
    // A replayed event from before the reset doesn't bring the run back.
    expect(play([{ ...before.runs.r1!.events[0]! }], after).order).toEqual([]);
  });
});

describe("an API restart", () => {
  it("starts the console over from the new process's log", () => {
    const at = (s: number, boot: string, runId: string): RunEvent =>
      ({ ...ev({ type: "started" }, runId), seq: s, boot }) as RunEvent;
    let state: RunsState = initialRuns;
    for (let s = 0; s <= 40; s++) state = runsReducer(state, { type: "event", event: at(s, "old", `r${s}`) });
    expect(state.lastSeq).toBe(40);
    // The new process's log starts again at 0: nothing is dropped as stale.
    state = runsReducer(state, { type: "event", event: at(0, "new", "fresh") });
    expect(state.order).toEqual(["fresh"]);
    expect(state.lastSeq).toBe(0);
    expect(state.boot).toBe("new");
  });
});
