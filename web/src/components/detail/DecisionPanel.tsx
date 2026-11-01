import { CheckCircleIcon, HourglassMediumIcon, type Icon, ProhibitIcon } from "@phosphor-icons/react";
import { type ReactNode, useState } from "react";

import { api, ApiError } from "../../api/client";
import type { Decision, Summary } from "../../api/types";
import { money } from "../../lib/format";
import { sentences } from "../../lib/text";
import { PixelIcon } from "../Pixel";

const VERDICT: Record<Decision, { label: string; icon: Icon }> = {
  approve: { label: "Approve", icon: CheckCircleIcon },
  reject: { label: "Reject", icon: ProhibitIcon },
  escalate: { label: "Escalate", icon: HourglassMediumIcon },
};

/** The agent's recommendation as one solid pill. */
function Verdict({ decision }: { decision: Decision }) {
  const { label, icon: Glyph } = VERDICT[decision];
  return (
    <span className="verdict">
      <Glyph size={18} weight="bold" aria-hidden="true" />
      {label}
    </span>
  );
}

/** A labelled inner panel: separates one kind of content from the next without rules. */
function Sheet({ label, children }: { label: string; children: ReactNode }) {
  return (
    <section className="sheet">
      <h4 className="sheet-label">{label}</h4>
      {children}
    </section>
  );
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

const TOKEN_KEY = "acme.vpToken";

/** A VP token, kept for this tab only, so closing it signs the VP out. */
function storedToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

function storeToken(token: string): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // Private windows may refuse storage; the token then lasts for this card.
  }
}

/** The VP's approve/reject form for a run paused at human_review. */
export function DecisionPanel({
  runId,
  summary,
  needsToken = false,
}: {
  runId: string;
  summary: Summary;
  /** The API takes decisions only with a VP token (VP_TOKENS is set). */
  needsToken?: boolean;
}) {
  const [note, setNote] = useState("");
  const [token, setToken] = useState(storedToken);
  // Also shown when the API turns a decision down for want of a token.
  const [refused, setRefused] = useState(false);
  const askToken = needsToken || refused;
  const [sending, setSending] = useState<"approve" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inv = summary.invoice;
  const reasons = summary.findings.filter((f) => f.severity !== "info");

  const decide = async (approved: boolean) => {
    setSending(approved ? "approve" : "reject");
    setError(null);
    try {
      await api.decide(runId, approved, note.trim(), token.trim() || undefined);
      storeToken(token.trim());
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        setRefused(true);
        storeToken("");
      }
      setError(e instanceof Error ? e.message : String(e));
      setSending(null);
    }
  };

  return (
    // The reasons scroll; the note and the buttons stay in view however small the card is.
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
        {summary.draft && (
          <div className="flex flex-wrap items-center gap-3">
            <span className="cap">agent recommends</span>
            <Verdict decision={summary.draft.decision} />
          </div>
        )}
        {summary.route_reason && (
          <Sheet label="why it needs a VP">
            <p className="m-0 font-medium">{capitalize(summary.route_reason)}.</p>
          </Sheet>
        )}
        {summary.draft && (
          <Sheet label="the agent's reasoning">
            <ul className="facts">
              {sentences(summary.draft.rationale).map((line, i) => (
                <li key={i}>
                  <PixelIcon node="dot" size={12} className="mt-1.5" />
                  <span>{line}</span>
                </li>
              ))}
            </ul>
          </Sheet>
        )}
        {reasons.length > 0 && (
          <Sheet label="flags to weigh">
            <ul className="facts">
              {reasons.map((f) => (
                <li key={f.code + f.detail}>
                  <span className="code-chip">{f.code}</span>
                  <span>{f.detail}</span>
                </li>
              ))}
            </ul>
          </Sheet>
        )}
      </div>
      {askToken && (
        <label className="flex flex-none flex-col gap-1">
          <span className="cap">your VP token (the log records your name)</span>
          <input
            className="key min-h-10 px-3 text-sm shadow-none active:translate-y-0"
            type="password"
            autoComplete="current-password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
          />
        </label>
      )}
      <label className="flex flex-none flex-col gap-1">
        <span className="cap">note for the audit log (optional)</span>
        <input
          className="key min-h-10 px-3 text-sm shadow-none active:translate-y-0"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          maxLength={500}
        />
      </label>
      <div className="grid flex-none grid-cols-1 gap-2 sm:grid-cols-2">
        <button type="button" className="key key-big key-go min-h-16" disabled={sending !== null} onClick={() => decide(true)}>
          <span className="text-lg font-medium">{sending === "approve" ? "Approving…" : "Approve and pay"}</span>
          <span className="mono text-xs">{inv ? `${money(inv.total, inv.currency)} via mock_payment` : "via mock_payment"}</span>
        </button>
        <button type="button" className="key key-big min-h-16" disabled={sending !== null} onClick={() => decide(false)}>
          <span className="text-lg font-medium">{sending === "reject" ? "Rejecting…" : "Reject"}</span>
          <span className="mono text-xs">logs your reason, no payment</span>
        </button>
      </div>
      {error && (
        <p role="alert" className="m-0 text-sm">
          <span className="tag mr-2">error</span>
          {error}
        </p>
      )}
    </div>
  );
}
