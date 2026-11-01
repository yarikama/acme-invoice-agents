import { pipeline, traceLines } from "../../state/pipeline";
import type { RunView } from "../../state/runs";
import { Pipeline } from "./Pipeline";
import { TraceScreen } from "./TraceScreen";

/** Pipeline lights over the live trace; the trace takes whatever height is left. */
export function AgentStream({ run, llm }: { run: RunView; llm: boolean }) {
  const label = run.summary?.invoice?.invoice_number ?? run.source;
  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <Pipeline segments={pipeline(run)} />
      <TraceScreen label={label} lines={traceLines(run, llm)} live={run.status === "running"} />
    </div>
  );
}
