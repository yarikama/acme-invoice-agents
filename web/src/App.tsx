import { useCallback, useMemo, useState } from "react";
import type { Layout } from "react-grid-layout";

import { Dashboard } from "./components/Dashboard";
import { Header } from "./components/Header";
import { Toolbar } from "./components/Toolbar";
import type { RunOverview } from "./api/types";
import { type Filter, matches } from "./lib/filters";
import { clearLayout, DEFAULT_LAYOUT, loadLayout, saveLayout } from "./state/layout";
import { useConsole } from "./state/useConsole";

export default function App() {
  const { state, stream, provider, vpAuth, snapshot, loaded } = useConsole();
  const [filter, setFilter] = useState<Filter>("all");
  const [picked, setPicked] = useState<string | null>(null);
  const [layout, setLayout] = useState<Layout>(loadLayout);

  const runs = useMemo(() => state.order.map((id) => state.runs[id]!), [state]);
  const visible = useMemo(() => runs.filter((r) => matches(filter, r.status)), [runs, filter]);

  // Until the user picks one, show the run that needs a decision, else the newest.
  const selected =
    (picked ? state.runs[picked] : undefined) ?? runs.find((r) => r.status === "awaiting_vp") ?? runs[0] ?? null;

  const onStarted = useCallback(
    (started: RunOverview[]) => {
      snapshot(started);
      if (started.length === 1) setPicked(started[0]!.id);
    },
    [snapshot],
  );
  const onSelect = useCallback((id: string) => setPicked(id), []);
  const onLayout = useCallback((next: Layout) => {
    setLayout(next);
    saveLayout(next);
  }, []);
  const onResetLayout = useCallback(() => {
    clearLayout();
    setLayout(DEFAULT_LAYOUT);
  }, []);
  const waiting = runs.filter((r) => r.status === "awaiting_vp").length;

  return (
    <div className="min-h-dvh">
      <Header stream={stream} onResetLayout={onResetLayout} />
      <main className="flex w-full flex-col gap-6 px-4 py-6 sm:px-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="display m-0 text-[clamp(44px,5.2vw,76px)] leading-[1.02]">Invoice inbox</h1>
            <p className="m-0 mt-2 text-lg text-ink-2">
              {!loaded && runs.length === 0
                ? "Loading invoices…"
                : runs.length === 0
                  ? "No invoices yet. Run the samples or upload one."
                  : waiting > 0
                    ? `${waiting} waiting for a VP decision.`
                    : "Nothing waiting for a decision."}
            </p>
          </div>
          <Toolbar onStarted={onStarted} />
        </div>

        <Dashboard
          runs={runs}
          visible={visible}
          loaded={loaded}
          filter={filter}
          onFilter={setFilter}
          selected={selected}
          onSelect={onSelect}
          llm={provider !== null && provider !== "offline"}
          vpAuth={vpAuth}
          layout={layout}
          onLayout={onLayout}
        />
      </main>
    </div>
  );
}
