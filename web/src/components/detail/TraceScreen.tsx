import { useEffect, useRef } from "react";

import type { TraceLine } from "../../state/pipeline";

const VOICE_CLASS = { model: "n-model", code: "n-code", human: "n-human" } as const;

/** The LCD: the agent trace, newest line kept in view while a run streams. */
export function TraceScreen({ label, lines, live }: { label: string; lines: TraceLine[]; live: boolean }) {
  const end = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (live) end.current?.scrollIntoView({ block: "nearest" });
  }, [lines.length, live]);

  return (
    <div className="lcd flex min-h-40 flex-1 flex-col" role="log" aria-label={`Agent trace for ${label}`} tabIndex={0}>
      <div className="mb-2 flex flex-wrap justify-between gap-x-4 border-b border-[#2c2c2a] pb-2 text-[#a3a39c]">
        <span>trace / {label}</span>
        <span className={live ? "n-human" : undefined}>{live ? "streaming" : "idle"}</span>
      </div>
      <ol className="m-0 flex min-h-0 flex-1 list-none flex-col gap-1 overflow-y-auto p-0">
        {lines.map((l, i) => (
          <li key={l.seq} className="grid grid-cols-[24px_96px_minmax(0,1fr)] gap-2">
            <span className="text-[#8e8e87]">{String(i + 1).padStart(2, "0")}</span>
            <span className={VOICE_CLASS[l.voice]}>{l.node}</span>
            <span>{l.text}</span>
          </li>
        ))}
        {live && (
          <li ref={end} className="grid grid-cols-[24px_96px_minmax(0,1fr)] gap-2 n-human">
            <span />
            <span>running</span>
            <span>
              <span className="cursor-blink" aria-hidden="true">
                _
              </span>
            </span>
          </li>
        )}
        {lines.length === 0 && !live && <li className="text-[#a3a39c]">No trace yet.</li>}
      </ol>
      <div className="mt-2 flex flex-wrap gap-x-4 border-t border-[#2c2c2a] pt-2 text-[#a3a39c]">
        <span className="n-model">blue: model</span>
        <span className="n-code">green: code</span>
        <span className="n-human">orange: human or live</span>
      </div>
    </div>
  );
}
