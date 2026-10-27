import { type Filter, matches, SEGMENTS } from "../lib/filters";
import { money } from "../lib/format";
import type { RunView } from "../state/runs";

function usdTotal(runs: RunView[]): string {
  const usd = runs.filter((r) => (r.summary?.invoice?.currency ?? "USD") === "USD");
  const sum = usd.reduce((acc, r) => acc + (r.summary?.invoice?.total ?? 0), 0);
  const other = runs.length - usd.length;
  return money(sum) + (other ? ` + ${other} non-USD` : "");
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
      className="grid grid-cols-[repeat(auto-fit,minmax(160px,1fr))] gap-px border border-seam bg-seam"
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
            className={`flex min-h-28 min-w-0 cursor-pointer flex-col gap-1 border-0 px-6 py-4 text-left text-ink hover:bg-raised ${
              selected ? "bracketed bg-raised" : "bg-panel"
            }`}
          >
            <span className="bk" aria-hidden="true" />
            <span className="flex items-center gap-2 text-sm font-medium">
              <span className={`lt ${seg.light}`} aria-hidden="true" />
              {seg.label}
            </span>
            <span className="text-4xl font-light">{members.length}</span>
            <span className="text-xs text-ink-2">{members.length ? usdTotal(members) : "none"}</span>
          </button>
        );
      })}
    </div>
  );
}
