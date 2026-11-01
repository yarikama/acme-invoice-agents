import { money } from "../lib/format";
import { dollars } from "../state/conversation";
import { duration, impact } from "../state/impact";
import type { RunView } from "../state/runs";

function Tile({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-[var(--r)] bg-[var(--tile)] px-4 py-3">
      <span className="text-sm font-medium">{label}</span>
      <span className="display text-3xl">{value}</span>
      <span className="text-xs text-ink-2">{note}</span>
    </div>
  );
}

/** The runs so far against the manual process in the brief. */
export function ImpactBank({ runs, replayed = false }: { runs: RunView[]; replayed?: boolean }) {
  const m = impact(runs);
  if (m.decided === 0) {
    return <p className="m-0 text-sm text-ink-2">Run the samples to see the time and money this saves.</p>;
  }
  const byHand = m.decided - m.noPerson;
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-3">
      <Tile
        label="Time to a decision"
        value={m.medianSeconds === null ? "n/a" : duration(m.medianSeconds)}
        note={
          replayed ? "replayed answers return at once; was 5 days by hand" : "median per invoice; was 5 days by hand"
        }
      />
      <Tile
        label="Settled with no person"
        value={`${m.noPerson} of ${m.decided}`}
        note={byHand ? `${byHand} sent to a VP, with the reasons` : "none needed a VP"}
      />
      <Tile
        label="Stopped before paying"
        value={money(m.stoppedUsd)}
        note={`${m.stopped} invoice${m.stopped === 1 ? "" : "s"} rejected, each with its reason`}
      />
      <Tile
        label="Model cost"
        value={m.costPerInvoice === null ? "n/a" : dollars(m.costPerInvoice)}
        note="per invoice, all agents together"
      />
    </div>
  );
}
