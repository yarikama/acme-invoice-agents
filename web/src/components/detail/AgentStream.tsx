import { activity, turns } from "../../state/conversation";
import { pipeline, startedAt, traceLines } from "../../state/pipeline";
import type { RunView } from "../../state/runs";
import { Conversation } from "./Conversation";
import { Pipeline } from "./Pipeline";

/** Pipeline lights over the run's conversation; the conversation takes the rest. */
export function AgentStream({ run, llm }: { run: RunView; llm: boolean }) {
  const label = run.summary?.invoice?.invoice_number ?? run.source;
  const segments = pipeline(run);
  const now = segments.find((s) => s.state === "run")?.key ?? null;
  return (
    <div className="flex h-full min-h-0 flex-col gap-5">
      <Pipeline segments={segments} />
      <Conversation
        key={run.id}
        label={label}
        turns={turns(traceLines(run, llm), startedAt(run))}
        live={run.status === "running"}
        doing={activity(now)}
      />
    </div>
  );
}
