import type { Finding, Invoice } from "../../api/types";
import { money } from "../../lib/format";

const key = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

/** The invoice's own lines, with the ones a finding names marked. */
export function LineItems({ invoice, findings }: { invoice: Invoice; findings: Finding[] }) {
  const flagged = new Set(
    findings
      .filter((f) => f.severity !== "info")
      .flatMap((f) => invoice.line_items.filter((li) => key(f.detail).includes(key(li.item))).map((li) => key(li.item))),
  );
  const cur = invoice.currency;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] border-collapse text-sm">
        <thead>
          <tr className="mono text-left text-xs text-ink-2">
            <th scope="col" className="border-b border-seam py-2 font-normal">item</th>
            <th scope="col" className="border-b border-seam py-2 text-right font-normal">qty</th>
            <th scope="col" className="border-b border-seam py-2 text-right font-normal">unit</th>
            <th scope="col" className="border-b border-seam py-2 text-right font-normal">amount</th>
          </tr>
        </thead>
        <tbody>
          {invoice.line_items.map((li, i) => {
            const bad = flagged.has(key(li.item));
            return (
              <tr key={i}>
                <td className="border-b border-seam py-2">
                  <span className="inline-flex items-center gap-2">
                    {bad && <span className="lt lt-awaiting_vp" aria-label="flagged" />}
                    <span className="mono text-xs">{li.item}</span>
                    {li.note && <span className="text-ink-2">{li.note}</span>}
                  </span>
                </td>
                <td className="border-b border-seam py-2 text-right">{li.quantity}</td>
                <td className="border-b border-seam py-2 text-right">{money(li.unit_price, cur)}</td>
                <td className="border-b border-seam py-2 text-right">{money(li.amount ?? li.quantity * li.unit_price, cur)}</td>
              </tr>
            );
          })}
        </tbody>
        <tfoot className="text-right">
          {invoice.subtotal != null && <Row label="subtotal" value={money(invoice.subtotal, cur)} />}
          {invoice.tax_amount != null && <Row label="tax" value={money(invoice.tax_amount, cur)} />}
          {invoice.shipping != null && <Row label="shipping" value={money(invoice.shipping, cur)} />}
          <Row label="total as printed" value={money(invoice.total, cur)} strong />
        </tfoot>
      </table>
    </div>
  );
}

function Row({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <tr>
      <td colSpan={3} className="cap py-1">
        {label}
      </td>
      <td className={`py-1 ${strong ? "font-semibold" : ""}`}>{value}</td>
    </tr>
  );
}
