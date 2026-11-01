import type { ReactNode } from "react";
import ReactGridLayout, { type Layout, useContainerWidth } from "react-grid-layout";

import { plural } from "../lib/format";
import { type Filter } from "../lib/filters";
import { SECTION_ICON, STATUS_ICON } from "../lib/icons";
import { COLS, GAP, ROW_HEIGHT, WIDGETS, type WidgetId } from "../state/layout";
import type { RunView } from "../state/runs";
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

export function Dashboard({
  runs,
  visible,
  loaded,
  filter,
  onFilter,
  selected,
  onSelect,
  llm,
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
  layout: Layout;
  onLayout: (layout: Layout) => void;
}) {
  const { width, containerRef, mounted } = useContainerWidth();
  const grid = mounted && width >= GRID_MIN_WIDTH;

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
          <DecisionPanel runId={selected.id} summary={s} />
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
          resizeConfig={{ enabled: true, handles: ["se"] }}
          onLayoutChange={onLayout}
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
