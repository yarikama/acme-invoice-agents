import { describe, expect, it } from "vitest";

import type { RunEvent, Summary } from "../api/types";
import { flowPaths, routeOf, settledTimeline } from "./routing";
import type { RunView } from "./runs";

const summary = (severities: ("error" | "warning" | "info")[], number = "INV-1"): Summary => ({
  status: "paid",
  source_format: "json",
  invoice: { invoice_number: number } as Summary["invoice"],
  findings: severities.map((severity, i) => ({ code: `C${i}`, severity, title: "t", detail: "d" })),
  route: "auto",
  draft: null,
  payment: null,
  reason: null,
  trace: [],
});

let seq = 0;
const ev = (partial: object, at: string): RunEvent =>
  ({ id: seq, seq: seq++, run_id: "r", source: "s", created_at: at, at, status: "running", ...partial }) as RunEvent;

function run(status: RunView["status"], s: Summary | null, events: RunEvent[] = [], id = "r"): RunView {
  return { id, source: "invoice.json", status, createdAt: "", summary: s, events, error: null };
}

describe("routeOf", () => {
  it("reads validation, who decided and the outcome", () => {
    const triaged = run("paid", summary([]), [
      ev(
        {
          type: "trace",
          node: "triage",
          stage: "triage",
          message:
            "m says approve (99% sure), suspicion 2%. Code confirmed every check passed, so the reasoning approver was skipped. Outcome: pay.",
        },
        "2026-11-01T00:00:01Z",
      ),
    ]);
    expect(routeOf(triaged)).toMatchObject({ validation: "clean", decider: "triage", outcome: "paid", label: "INV-1" });

    const vp = run("rejected", summary(["warning", "info"]), [
      ev({ type: "decision", approved: false, note: "" }, "2026-11-01T00:00:02Z"),
    ]);
    expect(routeOf(vp)).toMatchObject({ validation: "warnings", decider: "vp", outcome: "rejected" });

    expect(routeOf(run("awaiting_vp", summary(["error"])))).toMatchObject({
      validation: "errors",
      decider: "approver",
      outcome: "vp",
    });
  });

  it("skips runs that have not settled", () => {
    expect(routeOf(run("running", null))).toBeNull();
    expect(routeOf(run("failed", summary([])))).toBeNull();
  });
});

describe("flowPaths", () => {
  it("groups routes that took the same path", () => {
    const routes = [
      { id: "a", label: "A", validation: "clean", decider: "triage", outcome: "paid" },
      { id: "b", label: "B", validation: "clean", decider: "triage", outcome: "paid" },
      { id: "c", label: "C", validation: "errors", decider: "approver", outcome: "rejected" },
    ] as const;
    const paths = flowPaths([...routes]);
    expect(paths.map((p) => [p.key, p.routes.length])).toEqual([
      ["clean|triage|paid", 2],
      ["errors|approver|rejected", 1],
    ]);
  });
});

describe("settledTimeline", () => {
  it("times each settled run from the first start, in order", () => {
    const a = run(
      "paid",
      summary([]),
      [
        ev({ type: "started" }, "2026-11-01T00:00:00Z"),
        ev({ type: "done", summary: summary([]) }, "2026-11-01T00:00:30Z"),
      ],
      "a",
    );
    const b = run(
      "awaiting_vp",
      summary(["warning"]),
      [
        ev({ type: "started" }, "2026-11-01T00:00:10Z"),
        ev({ type: "awaiting_vp", request: null, summary: summary([]) }, "2026-11-01T00:00:20Z"),
      ],
      "b",
    );
    const c = run("running", null, [ev({ type: "started" }, "2026-11-01T00:00:40Z")], "c");
    expect(settledTimeline([a, b, c]).map((s) => [s.route.id, s.at])).toEqual([
      ["b", 20],
      ["a", 30],
    ]);
  });

  it("is empty without timestamps", () => {
    expect(settledTimeline([run("paid", summary([]))])).toEqual([]);
  });
});
