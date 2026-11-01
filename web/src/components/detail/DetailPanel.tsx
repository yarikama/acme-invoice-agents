import { money } from "../../lib/format";
import { pipeline, traceLines } from "../../state/pipeline";
import type { RunView } from "../../state/runs";
import { StatusLight } from "../StatusLight";
import { DecisionPanel } from "./DecisionPanel";
import { Findings } from "./Findings";
import { LineItems } from "./LineItems";
import { Pipeline } from "./Pipeline";
import { TraceScreen } from "./TraceScreen";

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

export function DetailPanel({ run, llm }: { run: RunView | null; llm: boolean }) {
  if (!run) {
    return (
      <section className="bg-panel p-6 text-ink-2">
        Select an invoice to see its pipeline, findings and agent trace.
      </section>
    );
  }
  const s = run.summary;
  const inv = s?.invoice;
  const label = inv?.invoice_number ?? run.source;
  const live = run.status === "running";

  return (
    <div className="flex flex-col gap-px bg-seam">
      <section aria-labelledby="detail-h" className="bg-panel p-6">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <span className="flex flex-wrap gap-2">
            <span className="tag">{label}</span>
            <span className="tag">{run.source}</span>
          </span>
          <span className="text-[15px] font-semibold">
            <StatusLight status={run.status} />
          </span>
        </div>
        <h2 id="detail-h" className="m-0 text-2xl font-medium">
          {inv?.vendor ?? (inv ? "No vendor" : run.source)}
        </h2>
        {inv && (
          <div className="mt-1 flex flex-wrap items-baseline gap-x-6 gap-y-1">
            <span className="text-4xl font-light">{money(inv.total, inv.currency)}</span>
            <span className="text-sm text-ink-2">
              Due {inv.due_date ?? "not stated"}
              {inv.payment_terms ? `, ${inv.payment_terms}` : ""}
            </span>
          </div>
        )}
        <p className="m-0 mt-3 text-sm text-ink-2">{closing(run)}</p>
      </section>

      {run.status === "awaiting_vp" && s && <DecisionPanel runId={run.id} summary={s} />}

      <section aria-labelledby="stream-h" className="flex flex-col gap-4 bg-panel p-6">
        <h3 id="stream-h" className="m-0 text-base font-medium">
          Agent stream
        </h3>
        <Pipeline segments={pipeline(run)} />
        <TraceScreen label={label} lines={traceLines(run, llm)} live={live} />
      </section>

      {s && <Findings findings={s.findings} />}
      {inv && s && inv.line_items.length > 0 && <LineItems invoice={inv} findings={s.findings} />}
      {/* Fills the column when the inbox beside it is taller. */}
      <div className="flex-1 bg-panel" />
    </div>
  );
}
