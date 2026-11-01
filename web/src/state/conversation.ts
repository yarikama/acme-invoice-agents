import type { Decision, StepUsage } from "../api/types";
import { splitToolLine } from "../lib/text";
import type { TraceLine } from "./pipeline";

/**
 * The trace as a conversation between the pipeline's actors, the way a chat
 * or an agent tracer shows it: one turn per actor, the approver's tool calls
 * nested inside its turn, each turn timed.
 */

export interface ToolUse {
  call: string;
  result: string;
}

interface Base {
  key: number;
  // Milliseconds this turn took: since the previous turn, or the run's start.
  ms?: number;
  // Tokens and cost of the model calls in this turn.
  usage?: StepUsage;
}

export type Turn = Base &
  (
    | { kind: "step"; node: "ingest" | "pay" | "reject" | "error" | string; text: string }
    | {
        kind: "validate";
        errors: number | null;
        warnings: number | null;
        route: string | null;
        reason: string | null;
        text: string;
      }
    | {
        kind: "triage";
        decision: Decision | null;
        sure: number | null;
        suspicion: number | null;
        // Triage settled the invoice itself, skipping the approver.
        settled: boolean;
        text: string;
      }
    | {
        kind: "approver";
        tools: ToolUse[];
        // Decisions the linter sent back, in order.
        sendbacks: string[];
        round: number | null;
        decision: Decision | null;
        rationale: string;
      }
    | { kind: "critic"; agrees: boolean; issues: string; outcome: string | null }
    | { kind: "human"; approved: boolean; by: string | null; note: string }
  );

const DECISION = "(approve|reject|escalate)";

export function parseValidate(text: string) {
  const m = /^(\d+) errors?, (\d+) warnings?; policy route: (\w+)(?: \((.*)\))?\.?$/.exec(text);
  return {
    errors: m ? Number(m[1]) : null,
    warnings: m ? Number(m[2]) : null,
    route: m ? m[3]! : null,
    reason: m?.[4] ?? null,
  };
}

export function parseTriage(text: string) {
  const m = new RegExp(`says ${DECISION} \\((\\d+)% sure\\), suspicion (\\d+)%`).exec(text);
  return {
    decision: (m?.[1] as Decision | undefined) ?? null,
    sure: m ? Number(m[2]) : null,
    suspicion: m ? Number(m[3]) : null,
    settled: /approver was skipped/.test(text),
  };
}

export function parseDraft(text: string) {
  const m = new RegExp(`^Draft (\\d+): ${DECISION}\\.\\s*([\\s\\S]*)$`).exec(text);
  return m
    ? { round: Number(m[1]), decision: m[2] as Decision, rationale: m[3]!.trim() }
    : { round: null, decision: null, rationale: text };
}

export function parseCritique(text: string) {
  const outcome = /Outcome: (\w+)\./.exec(text)?.[1] ?? null;
  const body = text.replace(/\s*Outcome: \w+\.\s*$/, "");
  const agrees = body.startsWith("Critic agrees");
  return { agrees, issues: agrees ? "" : body.replace(/^Critic objects:\s*/, ""), outcome };
}

export function turns(lines: TraceLine[], startedAt?: number): Turn[] {
  const out: Turn[] = [];
  let tools: ToolUse[] = [];
  let sendbacks: string[] = [];
  let last = startedAt;
  const timed = (line: TraceLine): Base => {
    const ms = line.at !== undefined && last !== undefined ? Math.max(0, line.at - last) : undefined;
    if (line.at !== undefined) last = line.at;
    return { key: line.seq, ms, usage: line.usage };
  };
  // Tool calls and send-backs arrive just before the draft they led to.
  const flush = (line: TraceLine) => {
    if (tools.length || sendbacks.length) {
      out.push({ ...timed(line), kind: "approver", tools, sendbacks, round: null, decision: null, rationale: "" });
      tools = [];
      sendbacks = [];
    }
  };

  for (const line of lines) {
    if (line.node === "tool") {
      tools.push(splitToolLine(line.text));
      continue;
    }
    if (line.node === "guardrail") {
      sendbacks.push(line.text.replace(/^Sent the decision back\.\s*/, ""));
      continue;
    }
    if (line.node === "review") {
      out.push({ ...timed(line), kind: "approver", tools, sendbacks, ...parseDraft(line.text) });
      tools = [];
      sendbacks = [];
      continue;
    }
    flush(line);
    if (line.node === "validate") out.push({ ...timed(line), kind: "validate", text: line.text, ...parseValidate(line.text) });
    else if (line.node === "triage") out.push({ ...timed(line), kind: "triage", text: line.text, ...parseTriage(line.text) });
    else if (line.node === "critique") out.push({ ...timed(line), kind: "critic", ...parseCritique(line.text) });
    else if (line.node === "human_review") {
      const m = /^VP (approved|rejected)(?: \(([^)]*)\))?\.\s*(?:Note:\s*)?([\s\S]*)$/.exec(line.text);
      out.push({
        ...timed(line),
        kind: "human",
        approved: m?.[1] === "approved",
        by: m?.[2] ?? null,
        note: m?.[3]?.trim() ?? "",
      });
    } else out.push({ ...timed(line), kind: "step", node: line.node, text: line.text });
  }
  const tail = lines.at(-1);
  if (tail) flush(tail);
  return out;
}

/** A run's totals so far: tokens, cost (if every step was priced) and time. */
export function totals(turns: Turn[]): { tokens: number; cost: number | null; ms: number } {
  let tokens = 0;
  let cost: number | null = 0;
  let ms = 0;
  for (const t of turns) {
    ms += t.ms ?? 0;
    if (!t.usage) continue;
    tokens += t.usage.input_tokens + t.usage.output_tokens;
    cost = cost === null || t.usage.cost_usd === null ? null : cost + t.usage.cost_usd;
  }
  return { tokens, cost, ms };
}

/** "2.1k tokens", "840 tokens". */
export function tokenWord(n: number): string {
  return `${n >= 1000 ? `${(n / 1000).toFixed(1)}k` : n} tokens`;
}

/** "$0.0042" below a cent, "$0.12" above. */
export function dollars(usd: number): string {
  return usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`;
}

/** What the run is doing now, in words, for the "thinking" line. */
export function activity(segment: string | null): string {
  switch (segment) {
    case "ingest":
      return "Reading the invoice";
    case "validate":
      return "Checking the rules";
    case "draft":
      return "The approver is reviewing";
    case "critique":
      return "The critic is checking the draft";
    case "pay":
      return "Paying";
    default:
      return "Working";
  }
}
