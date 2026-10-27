import { useCallback, useMemo, useState } from "react";

import { Header } from "./components/Header";
import { InboxTable } from "./components/InboxTable";
import { StatusBank } from "./components/StatusBank";
import { Toolbar } from "./components/Toolbar";
import { DetailPanel } from "./components/detail/DetailPanel";
import type { RunOverview } from "./api/types";
import { type Filter, matches } from "./lib/filters";
import { useConsole } from "./state/useConsole";

export default function App() {
  const { state, stream, provider, snapshot } = useConsole();
  const [filter, setFilter] = useState<Filter>("all");
  const [picked, setPicked] = useState<string | null>(null);

  const runs = useMemo(() => state.order.map((id) => state.runs[id]!), [state]);
  const visible = useMemo(() => runs.filter((r) => matches(filter, r.status)), [runs, filter]);

  // Until the user picks one, show the run that needs a decision, else the newest.
  const selected =
    (picked ? state.runs[picked] : undefined) ??
    runs.find((r) => r.status === "awaiting_vp") ??
    runs[0] ??
    null;

  const onStarted = useCallback(
    (started: RunOverview[]) => {
      snapshot(started);
      if (started.length === 1) setPicked(started[0]!.id);
    },
    [snapshot],
  );
  const onSelect = useCallback((id: string) => setPicked(id), []);
  const waiting = runs.filter((r) => r.status === "awaiting_vp").length;

  return (
    <div className="min-h-dvh">
      <Header stream={stream} provider={provider} />
      <main className="mx-auto flex max-w-[1440px] flex-col gap-6 px-4 py-6 sm:px-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="m-0 text-3xl font-medium">Invoice inbox</h1>
            <p className="m-0 mt-1 text-ink-2">
              {runs.length === 0
                ? "Nothing processed yet."
                : waiting > 0
                  ? `${waiting} waiting for a VP decision.`
                  : "Nothing waiting for a decision."}
            </p>
          </div>
          <Toolbar onStarted={onStarted} />
        </div>

        <StatusBank runs={runs} filter={filter} onFilter={setFilter} />

        <div className="grid grid-cols-1 gap-px bg-seam lg:grid-cols-[minmax(0,1fr)_minmax(0,560px)]">
          <section aria-labelledby="inbox-h" className="min-w-0 bg-panel">
            <div className="flex flex-wrap items-center justify-between gap-2 px-6 pt-6 pb-2">
              <h2 id="inbox-h" className="m-0 text-base font-medium">
                Invoices
              </h2>
              <span className="cap">
                showing {visible.length} of {runs.length}
              </span>
            </div>
            <InboxTable runs={visible} selectedId={selected?.id ?? null} onSelect={onSelect} />
          </section>
          <DetailPanel run={selected} llm={provider !== null && provider !== "offline"} />
        </div>
      </main>
    </div>
  );
}
