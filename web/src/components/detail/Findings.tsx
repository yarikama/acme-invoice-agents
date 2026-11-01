import type { Finding } from "../../api/types";

const SEVERITY_WORD = { error: "fail", warning: "flag", info: "note" } as const;
const ORDER = { error: 0, warning: 1, info: 2 } as const;

export function Findings({ findings }: { findings: Finding[] }) {
  const sorted = [...findings].sort((a, b) => ORDER[a.severity] - ORDER[b.severity]);
  if (findings.length === 0) return <p className="m-0 text-sm text-ink-2">Every check passed.</p>;
  return (
    <ul className="m-0 flex list-none flex-col gap-3 p-0">
      {sorted.map((f, i) => (
        <li key={`${f.code}-${i}`} className="flex items-start gap-4">
          <span className={`chip chip-${f.severity}`}>{SEVERITY_WORD[f.severity]}</span>
          <span className="flex min-w-0 flex-col">
            <span className="text-base font-medium">{f.title}</span>
            <span className="text-sm text-ink-2">{f.detail}</span>
            <span className="mono text-xs text-ink-2">{f.code}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}
