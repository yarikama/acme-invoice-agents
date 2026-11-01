import { describe, expect, it } from "vitest";

import { activity, parseCritique, parseDraft, parseTriage, parseValidate, turns } from "./conversation";
import type { TraceLine } from "./pipeline";

const line = (seq: number, node: string, text: string, at?: number): TraceLine => ({
  seq,
  node,
  voice: node === "tool" || node === "guardrail" ? "code" : "model",
  text,
  at,
});

describe("parsers", () => {
  it("reads the validation summary", () => {
    expect(parseValidate("0 errors, 1 warning; policy route: human (Billed in EUR).")).toEqual({
      errors: 0,
      warnings: 1,
      route: "human",
      reason: "Billed in EUR",
    });
    expect(parseValidate("2 errors, 0 warnings; policy route: reject.")).toMatchObject({ errors: 2, route: "reject", reason: null });
    expect(parseValidate("something else").errors).toBeNull();
  });

  it("reads triage, including when it settled the invoice", () => {
    expect(parseTriage("typesafe/jev-1.14 says escalate (100% sure), suspicion 11%. Needs the full review.")).toEqual({
      decision: "escalate",
      sure: 100,
      suspicion: 11,
      settled: false,
    });
    expect(parseTriage("m says approve (98% sure), suspicion 5%. Code confirmed every check passed, so the reasoning approver was skipped.").settled).toBe(true);
  });

  it("reads a draft and a critique", () => {
    expect(parseDraft("Draft 2: reject. Out of stock.\nTwice.")).toEqual({ round: 2, decision: "reject", rationale: "Out of stock.\nTwice." });
    expect(parseDraft("free text").decision).toBeNull();
    expect(parseCritique("Critic agrees. Outcome: human.")).toEqual({ agrees: true, issues: "", outcome: "human" });
    expect(parseCritique("Critic objects: The draft ignores OUT_OF_STOCK.")).toEqual({
      agrees: false,
      issues: "The draft ignores OUT_OF_STOCK.",
      outcome: null,
    });
  });
});

describe("turns", () => {
  it("nests tool calls and send-backs in the approver's turn and times each turn", () => {
    const got = turns(
      [
        line(1, "ingest", "Parsed XML in code.", 1_000),
        line(2, "validate", "0 errors, 1 warning; policy route: human (Billed in EUR).", 1_050),
        line(3, "tool", "convert_to_usd(4125.0, EUR): EUR 4,125.00 is $4,455.00.", 9_000),
        line(4, "guardrail", "Sent the decision back. Missing: FOREIGN_CURRENCY.", 9_000),
        line(5, "review", "Draft 1: approve. Fine.", 9_000),
        line(6, "critique", "Critic agrees. Outcome: human.", 12_500),
        line(7, "human_review", "VP approved. Note: ok", undefined),
      ],
      0,
    );
    expect(got.map((t) => t.kind)).toEqual(["step", "validate", "approver", "critic", "human"]);
    const approver = got[2]!;
    expect(approver).toMatchObject({
      kind: "approver",
      tools: [{ call: "convert_to_usd(4125.0, EUR)", result: "EUR 4,125.00 is $4,455.00." }],
      sendbacks: ["Missing: FOREIGN_CURRENCY."],
      round: 1,
      decision: "approve",
    });
    expect(got.map((t) => t.ms)).toEqual([1_000, 50, 7_950, 3_500, undefined]);
    expect(got[4]).toMatchObject({ kind: "human", approved: true, note: "ok" });
  });

  it("keeps tool calls that never reached a draft", () => {
    const got = turns([line(1, "tool", "check_inventory(X): X is not in the inventory database."), line(2, "error", "boom")]);
    expect(got.map((t) => t.kind)).toEqual(["approver", "step"]);
  });

  it("names what the run is doing", () => {
    expect(activity("draft")).toBe("The approver is reviewing");
    expect(activity(null)).toBe("Working");
  });
});
