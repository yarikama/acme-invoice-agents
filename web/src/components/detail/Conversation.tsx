import { useEffect, useRef, useState, type ReactNode } from "react";

import type { Decision, StepUsage } from "../../api/types";
import { sentences } from "../../lib/text";
import { dollars, tokenWord, totals, type Turn } from "../../state/conversation";
import type { TraceVoice } from "../../state/pipeline";
import { PixelIcon, PixelSpinner, Typewriter } from "../Pixel";

const VERDICT_WORD: Record<Decision, string> = { approve: "Approve", reject: "Reject", escalate: "Escalate" };

/** "x-ai/grok-4.7" -> "grok-4.7": the model a turn ran on, without its vendor. */
function shortModel(name: string): string {
  return name
    .split(",")
    .map((n) => n.trim().split("/").pop())
    .join(", ");
}

function seconds(ms: number | undefined): string | null {
  if (ms === undefined) return null;
  return ms < 1000 ? `${Math.max(ms, 1).toFixed(0)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

/** An actor's avatar: a pixel glyph on its voice's colour. */
function Avatar({ node, voice, active = false }: { node: string; voice: TraceVoice; active?: boolean }) {
  return (
    <span className={`avatar avatar-${voice} ${active ? "avatar-active" : ""}`} aria-hidden="true">
      <PixelIcon node={node} size={16} />
    </span>
  );
}

function TurnFrame({
  node,
  voice,
  name,
  role,
  ms,
  usage,
  fresh,
  children,
}: {
  node: string;
  voice: TraceVoice;
  name: string;
  role: string;
  ms?: number;
  usage?: StepUsage;
  fresh: boolean;
  children: ReactNode;
}) {
  const took = seconds(ms);
  // What the step cost, the way a tracer shows it: tokens, then dollars if priced.
  const spent = usage
    ? [tokenWord(usage.input_tokens + usage.output_tokens), usage.cost_usd !== null ? dollars(usage.cost_usd) : null]
        .filter(Boolean)
        .join(" · ")
    : null;
  return (
    <li className={`turn ${fresh ? "turn-new" : ""}`}>
      <Avatar node={node} voice={voice} />
      <div className="flex min-w-0 flex-col gap-2">
        <div className="flex items-baseline gap-2">
          <span className="font-medium">{name}</span>
          <span className={`role role-${voice}`}>
            {voice === "model" && usage?.model ? shortModel(usage.model) : role}
          </span>
          <span className="mono ml-auto flex gap-3 text-xs text-ink-2">
            {spent && <span>{spent}</span>}
            {took && <span>{took}</span>}
          </span>
        </div>
        {children}
      </div>
    </li>
  );
}

function Pill({ decision }: { decision: Decision }) {
  return <span className={`pill pill-${decision}`}>{VERDICT_WORD[decision]}</span>;
}

function Stat({ children }: { children: ReactNode }) {
  return <span className="stat">{children}</span>;
}

/** The reasoning, one sentence a line; long ones fold after three. */
function Reasoning({ text, fresh }: { text: string; fresh: boolean }) {
  const all = sentences(text);
  const [open, setOpen] = useState(false);
  const shown = open ? all : all.slice(0, 3);
  return (
    <div className="bubble">
      <ul className="facts">
        {shown.map((s, i) => (
          <li key={i}>
            <PixelIcon node="dot" size={12} className="mt-1.5 text-ink-2" />
            <span>
              <Typewriter text={s} animate={fresh} />
            </span>
          </li>
        ))}
      </ul>
      {all.length > 3 && (
        <button type="button" className="linkish" onClick={() => setOpen((o) => !o)}>
          {open ? "Show less" : `Show all ${all.length} points`}
        </button>
      )}
    </div>
  );
}

function TurnView({ turn, fresh }: { turn: Turn; fresh: boolean }) {
  switch (turn.kind) {
    case "validate":
      return (
        <TurnFrame node="validate" voice="code" name="Validation" role="code" ms={turn.ms} usage={turn.usage} fresh={fresh}>
          {turn.errors === null ? (
            <p className="m-0">{turn.text}</p>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <Stat>{turn.errors} {turn.errors === 1 ? "error" : "errors"}</Stat>
              <Stat>{turn.warnings} {turn.warnings === 1 ? "warning" : "warnings"}</Stat>
              <Stat>route: {turn.route}</Stat>
              {turn.reason && <span className="text-sm text-ink-2">{turn.reason}</span>}
            </div>
          )}
        </TurnFrame>
      );
    case "triage":
      return (
        <TurnFrame node="triage" voice="model" name="Triage" role="decision model" ms={turn.ms} usage={turn.usage} fresh={fresh}>
          {turn.decision ? (
            <div className="flex flex-wrap items-center gap-2">
              <Pill decision={turn.decision} />
              <Stat>{turn.sure}% sure</Stat>
              <Stat>suspicion {turn.suspicion}%</Stat>
              <span className="text-sm text-ink-2">
                {turn.settled ? "Code confirmed it, so the approver was skipped." : "Handed to the approver."}
              </span>
            </div>
          ) : (
            <p className="m-0">{turn.text}</p>
          )}
        </TurnFrame>
      );
    case "approver":
      return (
        <TurnFrame node="review" voice="model" name="Approver" role="model" ms={turn.ms} usage={turn.usage} fresh={fresh}>
          {turn.tools.map((t, i) => (
            <div key={i} className="tool-use" style={fresh ? { animationDelay: `${i * 90}ms` } : undefined}>
              <PixelIcon node="tool" size={14} className="mt-1 text-[var(--code)]" />
              <div className="min-w-0">
                <div className="mono text-sm">{t.call}</div>
                {t.result && <div className="text-sm text-ink-2">{t.result}</div>}
              </div>
            </div>
          ))}
          {turn.sendbacks.map((s, i) => (
            <div key={`g${i}`} className="sendback">
              <PixelIcon node="guardrail" size={14} className="mt-1" />
              <div className="min-w-0 text-sm">
                <span className="font-medium">Code sent the decision back. </span>
                {s}
              </div>
            </div>
          ))}
          {turn.decision && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-ink-2">Draft {turn.round}</span>
              <Pill decision={turn.decision} />
            </div>
          )}
          {turn.rationale && <Reasoning text={turn.rationale} fresh={fresh} />}
        </TurnFrame>
      );
    case "critic":
      return (
        <TurnFrame node="critique" voice="model" name="Critic" role="model" ms={turn.ms} usage={turn.usage} fresh={fresh}>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`pill ${turn.agrees ? "pill-approve" : "pill-reject"}`}>{turn.agrees ? "Agrees" : "Objects"}</span>
            {turn.outcome && <Stat>outcome: {turn.outcome}</Stat>}
          </div>
          {turn.issues && <div className="bubble text-sm">{turn.issues}</div>}
        </TurnFrame>
      );
    case "human":
      return (
        <TurnFrame node="human_review" voice="human" name="VP" role="human" ms={turn.ms} usage={turn.usage} fresh={fresh}>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`pill ${turn.approved ? "pill-approve" : "pill-reject"}`}>{turn.approved ? "Approved" : "Rejected"}</span>
            {turn.note && <span className="text-sm">{turn.note}</span>}
          </div>
        </TurnFrame>
      );
    case "step": {
      const name = { ingest: "Ingest", pay: "Payment", reject: "Not paid", error: "Run failed" }[turn.node] ?? turn.node;
      return (
        <TurnFrame node={turn.node} voice="code" name={name} role="code" ms={turn.ms} usage={turn.usage} fresh={fresh}>
          <p className="m-0 text-[var(--ink)]">{turn.text}</p>
        </TurnFrame>
      );
    }
  }
}

/** The run so far, in one line: model tokens, cost and time. */
function RunTotals({ turns, live }: { turns: Turn[]; live: boolean }) {
  const t = totals(turns);
  if (turns.length === 0) return null;
  const parts = [
    t.tokens ? tokenWord(t.tokens) : "no model calls",
    t.tokens && t.cost !== null ? dollars(t.cost) : null,
    t.ms ? seconds(t.ms) : null,
  ].filter(Boolean);
  return (
    <p className="m-0 mb-4 flex flex-wrap items-center gap-2 text-sm text-ink-2">
      <span className="stat">{live ? "so far" : "this run"}</span>
      {parts.join(" · ")}
    </p>
  );
}

/**
 * The agent run as a conversation: who acted, what they looked up, what
 * they decided, and how long each took. While the run streams, a shimmering
 * line says what is happening now.
 */
export function Conversation({ label, turns, live, doing }: { label: string; turns: Turn[]; live: boolean; doing: string }) {
  const end = useRef<HTMLDivElement>(null);
  // Turns present on first render are history; only later ones animate.
  // The parent keys this component by run, so switching runs starts fresh.
  const [lastSeen] = useState(() => turns.at(-1)?.key ?? -1);
  useEffect(() => {
    if (live) end.current?.scrollIntoView({ block: "nearest" });
  }, [turns.length, live]);

  return (
    <div className="flex min-h-0 flex-1 flex-col" role="log" aria-label={`Agent run for ${label}`}>
      <RunTotals turns={turns} live={live} />
      <ol className="m-0 flex min-h-0 flex-1 list-none flex-col gap-5 overflow-y-auto p-0 pr-1">
        {turns.map((t) => (
          <TurnView key={t.key} turn={t} fresh={t.key > lastSeen} />
        ))}
        {turns.length === 0 && !live && <li className="text-ink-2">Nothing has run yet.</li>}
      </ol>
      {live && (
        <div ref={end} className="turn mt-5">
          <span className="avatar avatar-model avatar-active" aria-hidden="true">
            <PixelSpinner />
          </span>
          <span className="shimmer self-center">{doing}…</span>
        </div>
      )}
    </div>
  );
}
