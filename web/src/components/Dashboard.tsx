import { type ReactNode, useState } from "react";
import ReactGridLayout, { type Layout, useContainerWidth } from "react-grid-layout";

import { plural } from "../lib/format";
import { type Filter } from "../lib/filters";
import { SECTION_ICON, STATUS_ICON } from "../lib/icons";
import { COLS, GAP, ROW_HEIGHT, WIDGETS, type WidgetId } from "../state/layout";
import {
  DECIDER_LABEL,
  flowPaths,
  OUTCOME_LABEL,
  routeOf,
  type Route,
  settledTimeline,
  VALIDATION_LABEL,
} from "../state/routing";
import type { RunView } from "../state/runs";
import { RoutingFlow } from "./charts/RoutingFlow";
import { Throughput } from "./charts/Throughput";
import { AgentStream } from "./detail/AgentStream";
import { DecisionPanel } from "./detail/DecisionPanel";
import { Findings } from "./detail/Findings";
import { InvoiceSummary } from "./detail/InvoiceSummary";
import { LineItems } from "./detail/LineItems";
import { InboxTable } from "./InboxTable";
import { StatusBank } from "./StatusBank";
import { StatusLight } from "./StatusLight";
import { Widget } from "./Widget";

// Below this container width (a 900px window less the page's 24px gutters)
// the widgets stack in one column at their natural height.
const GRID_MIN_WIDTH = 852;

function Muted({ children }: { children: string }) {
  return <p className="m-0 text-sm text-ink-2">{children}</p>;
}

function Legend() {
  return (
    <div className="chart-legend">
      {(["paid", "vp", "rejected"] as const).map((o) => (
        <span key={o}>
          <i style={{ background: `var(--flow-${o})` }} />
          {OUTCOME_LABEL[o]}
        </span>
      ))}
    </div>
  );
}

export function Dashboard({
  runs,
  visible,
  loaded,
  filter,
  onFilter,
  selected,
  onSelect,
  llm,
  vpAuth = false,
  layout,
  onLayout,
}: {
  runs: RunView[];
  visible: RunView[];
  loaded: boolean;
  filter: Filter;
  onFilter: (f: Filter) => void;
  selected: RunView | null;
  onSelect: (id: string) => void;
  llm: boolean;
  /** VP decisions need a token. */
  vpAuth?: boolean;
  layout: Layout;
  onLayout: (layout: Layout) => void;
}) {
  const { width, containerRef, mounted } = useContainerWidth();
  // True while a card is dragged or resized; panels drop their frosting meanwhile.
  const [moving, setMoving] = useState(false);
  const start = () => setMoving(true);
  const stop = () => setMoving(false);
  const grid = mounted && width >= GRID_MIN_WIDTH;

  const routes = runs.map(routeOf).filter((r): r is Route => r !== null);
  const paths = flowPaths(routes);
  const settled = settledTimeline(runs);

  const s = selected?.summary;
  const inv = s?.invoice;
  const waiting = selected?.status === "awaiting_vp";
  const errors = s?.findings.filter((f) => f.severity === "error").length ?? 0;

  const widgets: Record<WidgetId, (draggable: boolean) => ReactNode> = {
    overview: (d) => (
      <Widget id="overview" title="Overview" icon={STATUS_ICON.all} draggable={d}>
        <StatusBank runs={runs} filter={filter} onFilter={onFilter} />
      </Widget>
    ),
    routing: (d) => (
      <Widget
        id="routing"
        title="Routing"
        icon={SECTION_ICON.routing}
        draggable={d}
        aside={<span className="cap whitespace-nowrap">{routes.length} settled</span>}
      >
        {routes.length === 0 ? (
          <Muted>Run the samples to see where each invoice goes.</Muted>
        ) : (
          <div className="flex flex-col gap-3">
            <Legend />
            <RoutingFlow paths={paths} total={routes.length} />
            <details>
              <summary className="cursor-pointer text-sm text-ink-2">Show as a table</summary>
              <table className="chart-table">
                <thead>
                  <tr>
                    <th>Validation</th>
                    <th>Decided by</th>
                    <th>Outcome</th>
                    <th>Invoices</th>
                  </tr>
                </thead>
                <tbody>
                  {paths.map((p) => (
                    <tr key={p.key}>
                      <td>{VALIDATION_LABEL[p.steps[0]]}</td>
                      <td>{DECIDER_LABEL[p.steps[1]]}</td>
                      <td>{OUTCOME_LABEL[p.steps[2]]}</td>
                      <td>{p.routes.map((r) => r.label).join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </div>
        )}
      </Widget>
    ),
    throughput: (d) => (
      <Widget id="throughput" title="Settled over time" icon={SECTION_ICON.throughput} draggable={d}>
        {settled.length === 0 ? (
          <Muted>The line grows as invoices settle.</Muted>
        ) : (
          <div className="flex flex-col gap-3">
            <Legend />
            <Throughput settled={settled} />
            <details>
              <summary className="cursor-pointer text-sm text-ink-2">Show as a table</summary>
              <table className="chart-table">
                <thead>
                  <tr>
                    <th>Invoice</th>
                    <th>Outcome</th>
                    <th>Settled at</th>
                  </tr>
                </thead>
                <tbody>
                  {settled.map((x) => (
                    <tr key={x.route.id}>
                      <td>{x.route.label}</td>
                      <td>{OUTCOME_LABEL[x.route.outcome]}</td>
                      <td>{x.at < 60 ? `${x.at.toFixed(1)}s` : `${(x.at / 60).toFixed(1)} min`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </div>
        )}
      </Widget>
    ),
    inbox: (d) => (
      <Widget
        id="inbox"
        title="Invoices"
        icon={SECTION_ICON.invoices}
        flush
        draggable={d}
        aside={
          <span className="cap whitespace-nowrap">
            showing {visible.length} of {runs.length}
          </span>
        }
      >
        <InboxTable
          runs={visible}
          loaded={loaded}
          filtered={runs.length > 0}
          selectedId={selected?.id ?? null}
          onSelect={onSelect}
        />
      </Widget>
    ),
    summary: (d) => (
      <Widget
        id="summary"
        title="Invoice"
        icon={SECTION_ICON.summary}
        draggable={d}
        aside={selected && <StatusLight status={selected.status} />}
      >
        {selected ? <InvoiceSummary run={selected} /> : <Muted>Select an invoice to see it here.</Muted>}
      </Widget>
    ),
    decision: (d) => (
      <Widget id="decision" title="VP decision" icon={SECTION_ICON.decision} signal={waiting} draggable={d}>
        {waiting && selected && s ? (
          <DecisionPanel runId={selected.id} summary={s} needsToken={vpAuth} />
        ) : (
          <Muted>
            {selected ? "Nothing to decide: this invoice isn't waiting for a VP." : "Nothing selected."}
          </Muted>
        )}
      </Widget>
    ),
    stream: (d) => (
      <Widget id="stream" title="Agent stream" icon={SECTION_ICON.stream} draggable={d}>
        {selected ? <AgentStream run={selected} llm={llm} /> : <Muted>No invoice selected.</Muted>}
      </Widget>
    ),
    findings: (d) => (
      <Widget
        id="findings"
        title="Findings"
        icon={SECTION_ICON.findings}
        draggable={d}
        aside={
          s && (
            <span className="cap whitespace-nowrap">
              {s.findings.length ? `${plural(errors, "hard failure")}, ${s.findings.length} total` : "all passed"}
            </span>
          )
        }
      >
        {s ? <Findings findings={s.findings} /> : <Muted>Findings appear once validation runs.</Muted>}
      </Widget>
    ),
    lines: (d) => (
      <Widget id="lines" title="Line items" icon={SECTION_ICON.lineItems} draggable={d}>
        {inv && s && inv.line_items.length > 0 ? (
          <LineItems invoice={inv} findings={s.findings} />
        ) : (
          <Muted>No line items to show.</Muted>
        )}
      </Widget>
    ),
  };

  return (
    <div ref={containerRef}>
      {grid ? (
        <ReactGridLayout
          width={width}
          layout={layout}
          gridConfig={{ cols: COLS, rowHeight: ROW_HEIGHT, margin: [GAP, GAP], containerPadding: [0, 0] }}
          dragConfig={{ enabled: true, handle: ".widget-handle" }}
          resizeConfig={{ enabled: true, handles: ["e", "w", "s", "se", "sw"] }}
          onLayoutChange={onLayout}
          onDragStart={start}
          onDragStop={stop}
          onResizeStart={start}
          onResizeStop={stop}
          className={moving ? "grid-moving" : undefined}
        >
          {WIDGETS.map((id) => (
            <div key={id}>{widgets[id](true)}</div>
          ))}
        </ReactGridLayout>
      ) : (
        <div className="flex flex-col gap-4">
          {WIDGETS.map((id) => (
            <div key={id}>{widgets[id](false)}</div>
          ))}
        </div>
      )}
    </div>
  );
}
