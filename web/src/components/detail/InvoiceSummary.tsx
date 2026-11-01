import { money } from "../../lib/format";
import type { RunView } from "../../state/runs";

function closing(run: RunView): string {
  const s = run.summary;
  if (run.status === "paid" && s?.payment) return `Paid ${money(s.payment.amount, s.payment.currency)} through mock_payment.`;
  if (run.status === "rejected") return s?.reason ?? "Rejected.";
  if (run.status === "failed") return run.error ?? "The run failed.";
  if (run.status === "awaiting_vp") {
    if (s?.route_reason) return `Paused for a VP: ${s.route_reason}.`;
    if (s?.draft?.decision === "escalate") return "Paused for a VP: the agent escalated it.";
    return "Paused until a VP decides.";
  }
  if (run.status === "queued") return "Queued behind the runs before it.";
  return "Agents are working on it.";
}

/** Who billed what, and where the run stands. */
export function InvoiceSummary({ run }: { run: RunView }) {
  const inv = run.summary?.invoice;
  return (
    <div className="flex flex-col gap-1">
      <span className="flex flex-wrap gap-2">
        <span className="tag">{inv?.invoice_number ?? run.source}</span>
        <span className="tag">{run.source}</span>
      </span>
      <p className="m-0 mt-2 text-2xl font-medium">{inv?.vendor ?? (inv ? "No vendor" : run.source)}</p>
      {inv && (
        <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
          <span className="display text-4xl">{money(inv.total, inv.currency)}</span>
          <span className="text-sm text-ink-2">
            Due {inv.due_date ?? "not stated"}
            {inv.payment_terms ? `, ${inv.payment_terms}` : ""}
          </span>
        </div>
      )}
      <p className="m-0 mt-2 text-sm text-ink-2">{closing(run)}</p>
    </div>
  );
}
