import { type Filter, matches, SEGMENTS } from "../lib/filters";
import { STATUS_ICON } from "../lib/icons";
import { money } from "../lib/format";
import type { RunView } from "../state/runs";
import { Badge } from "./Badge";

// Totals only add up within one currency, so other currencies are counted, not summed.
function Total({ runs }: { runs: RunView[] }) {
  const priced = runs.filter((r) => r.summary?.invoice?.total != null);
  if (runs.length === 0) return <>none</>;
  if (priced.length === 0) return <>no amount</>;
  const usd = priced.filter((r) => r.summary!.invoice!.currency === "USD");
  const sum = usd.reduce((acc, r) => acc + r.summary!.invoice!.total!, 0);
  const other = priced.length - usd.length;
  if (usd.length === 0) return <span className="whitespace-nowrap">{other} non-USD</span>;
  return (
    <>
      <span className="whitespace-nowrap">{money(sum)}</span>
      {other > 0 && <span className="whitespace-nowrap"> + {other} non-USD</span>}
    </>
  );
}

export function StatusBank({
  runs,
  filter,
  onFilter,
}: {
  runs: RunView[];
  filter: Filter;
  onFilter: (f: Filter) => void;
}) {
  return (
    <div
      role="group"
      aria-label="Filter invoices by status"
      className="grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-3"
    >
      {SEGMENTS.map((seg) => {
        const members = runs.filter((r) => matches(seg.key, r.status));
        const selected = filter === seg.key;
        return (
          <button
            key={seg.key}
            type="button"
            aria-pressed={selected}
            onClick={() => onFilter(seg.key)}
            className={`flex min-w-0 cursor-pointer flex-col gap-1 rounded-[var(--r)] border border-seam bg-panel px-4 py-3 text-left font-[inherit] text-ink hover:bg-raised ${
              selected ? "card-on" : ""
            }`}
          >
            <span className="flex items-center gap-2.5 text-sm font-medium">
              {/* Orange only when something is actually waiting for a person. */}
              <Badge icon={STATUS_ICON[seg.key]} signal={seg.key === "awaiting_vp" && members.length > 0} />
              {seg.label}
            </span>
            <span className="text-3xl font-light">{members.length}</span>
            <span className="text-xs text-ink-2">
              <Total runs={members} />
            </span>
          </button>
        );
      })}
    </div>
  );
}
