import { useState } from "react";

import { api } from "../../api/client";
import type { Summary } from "../../api/types";
import { money } from "../../lib/format";

/** The VP's approve/reject form for a run paused at human_review. */
export function DecisionPanel({ runId, summary }: { runId: string; summary: Summary }) {
  const [note, setNote] = useState("");
  const [sending, setSending] = useState<"approve" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inv = summary.invoice;
  const reasons = summary.findings.filter((f) => f.severity !== "info");

  const decide = async (approved: boolean) => {
    setSending(approved ? "approve" : "reject");
    setError(null);
    try {
      await api.decide(runId, approved, note.trim());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSending(null);
    }
  };

  return (
    // The reasons scroll; the note and the buttons stay in view however small the card is.
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
        {summary.draft && (
          <p className="m-0">
            <span className="cap mr-2">agent recommends {summary.draft.decision}</span>
            {summary.draft.rationale}
          </p>
        )}
        {reasons.length > 0 && (
          <ul className="m-0 flex list-none flex-col gap-1 p-0 text-sm">
            {reasons.map((f) => (
              <li key={f.code + f.detail}>
                <span className="mono mr-2 text-xs text-ink-2">{f.code}</span>
                {f.detail}
              </li>
            ))}
          </ul>
        )}
      </div>
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
