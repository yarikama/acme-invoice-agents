import type { Finding, Invoice, RunEvent, StageOutput, StepUsage } from "../api/types";
import type { RunView } from "./runs";

export type LedState = "off" | "run" | "done" | "flag" | "block" | "fail" | "wait" | "skip";

export interface Segment {
  key: "ingest" | "validate" | "draft" | "critique" | "vp" | "pay";
  label: string;
  state: LedState;
  word: string;
}

// Graph node -> pipeline segment.
const NODE_SEGMENT: Record<string, Segment["key"]> = {
  ingest: "ingest",
  validate: "validate",
  // System 1 triage drafts the decision when it settles an invoice itself.
  triage: "draft",
  review: "draft",
  critique: "critique",
  human_review: "vp",
  pay: "pay",
  reject: "pay",
};

// Progress stage -> the segment a run is in while that stage runs.
const PROGRESS_SEGMENT: Record<string, Segment["key"]> = {
  ingest: "ingest",
  validate: "validate",
  pay: "pay",
};

const LABELS: Record<Segment["key"], string> = {
  ingest: "ingest",
  validate: "validate",
  draft: "draft",
  critique: "critique",
  vp: "vp",
  pay: "pay",
};

const WORDS: Record<LedState, string> = {
  off: "Not reached",
  run: "Running",
  done: "Done",
  flag: "Flagged",
  // Validation stopped the invoice. Not "Failed", which means the run itself broke.
  block: "Blocked",
  fail: "Failed",
  wait: "Waiting",
  skip: "Skipped",
};

function activeSegment(events: RunEvent[]): Segment["key"] | null {
  const last = events.at(-1);
  if (!last) return null;
  if (last.type === "progress") {
    if (last.stage === "approve") return last.step === "critique" ? "critique" : "draft";
    return PROGRESS_SEGMENT[last.stage] ?? null;
  }
  return null;
}

export function pipeline(run: RunView): Segment[] {
  const reached = new Set<Segment["key"]>();
  for (const e of run.events) {
    if (e.type === "trace" && NODE_SEGMENT[e.node]) reached.add(NODE_SEGMENT[e.node]!);
  }
  const findings = run.summary?.findings ?? [];
  const states: Record<Segment["key"], LedState> = {
    ingest: reached.has("ingest") ? "done" : "off",
    validate: !reached.has("validate")
      ? "off"
      : findings.some((f) => f.severity === "error")
        ? "block"
        : findings.some((f) => f.severity === "warning")
          ? "flag"
          : "done",
    draft: reached.has("draft") ? "done" : "off",
    critique: reached.has("critique") ? "done" : "off",
    vp: reached.has("vp") ? "done" : "off",
    pay: "off",
  };

  if (run.status === "awaiting_vp") states.vp = "wait";
  if (run.status === "paid") states.pay = "done";
  if (run.status === "rejected") states.pay = "skip";
  if (run.status === "failed") {
    const firstOff = (Object.keys(states) as Segment["key"][]).find((k) => states[k] === "off");
    if (firstOff) states[firstOff] = "fail";
  }
  // Reached in a finished run but never visited: e.g. no VP for small invoices.
  if (run.status === "paid" || run.status === "rejected") {
    if (states.vp === "off") states.vp = "skip";
    // Triage settled it, so there was no draft to critique.
    if (states.draft === "done" && states.critique === "off") states.critique = "skip";
  }
  if (run.status === "running") {
    const active = activeSegment(run.events);
    if (active) states[active] = "run";
  }

  return (Object.keys(states) as Segment["key"][]).map((key) => ({
    key,
    label: LABELS[key],
    state: states[key],
    word: WORDS[states[key]],
  }));
}

export type TraceVoice = "model" | "code" | "human";

export interface TraceLine {
  seq: number;
  node: string;
  voice: TraceVoice;
  text: string;
  // Epoch milliseconds when the line arrived, if the server said.
  at?: number;
  // Tokens and cost of the step this line ends, if it called a model.
  usage?: StepUsage;
  // What the step produced, when it says.
  output?: StageOutput;
}

// The VP's decision shows up once, as the human_review node's trace line.
export function traceLines(run: RunView, llm: boolean): TraceLine[] {
  const lines: TraceLine[] = [];
  for (const e of run.events) {
    if (e.type === "trace" && e.tool) {
      // The model asked; code answered. The result is a fact, so it reads as code.
      // A guardrail line is code sending the approver's decision back.
      const node = e.tool === "guardrail" ? "guardrail" : "tool";
      lines.push({ seq: e.seq, node, voice: "code", text: e.message, at: time(e.at) });
    } else if (e.type === "trace") {
      lines.push({
        seq: e.seq,
        node: e.node,
        voice: voiceOf(e.node, e.message, llm),
        text: e.message,
        at: time(e.at),
        usage: e.usage,
        output: e.data,
      });
    } else if (e.type === "run_error") {
      lines.push({ seq: e.seq, node: "error", voice: "code", text: e.message, at: time(e.at) });
    }
  }
  return lines;
}

function time(iso: string | undefined): number | undefined {
  const ms = iso ? Date.parse(iso) : NaN;
  return Number.isNaN(ms) ? undefined : ms;
}

/** When the run started, for timing its first step. */
export function startedAt(run: RunView): number | undefined {
  const first = run.events.find((e) => e.type === "started") ?? run.events[0];
  return time(first?.at);
}

function voiceOf(node: string, message: string, llm: boolean): TraceVoice {
  if (node === "human_review") return "human";
  // A decision model, whatever the approver's provider.
  if (node === "triage") return "model";
  if (node === "ingest") return message.includes("LLM") ? "model" : "code";
  // The approver and critic are models only when an LLM is configured.
  if (node === "review" || node === "critique") return llm ? "model" : "code";
  return "code";
}

/**
 * The invoice and its findings as far as the run has got: the final summary
 * once there is one, otherwise what ingest and validation reported on their
 * trace lines, so the panels fill in while the agents are still working.
 */
export function soFar(run: RunView): { invoice: Invoice | null; findings: Finding[] | null } {
  if (run.summary) return { invoice: run.summary.invoice, findings: run.summary.findings };
  let invoice: Invoice | null = null;
  let findings: Finding[] | null = null;
  for (const e of run.events) {
    if (e.type !== "trace" || !e.data) continue;
    if (e.data.invoice) invoice = e.data.invoice;
    if (e.data.findings) findings = e.data.findings;
  }
  return { invoice, findings };
}
