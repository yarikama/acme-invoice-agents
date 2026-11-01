import { useEffect, useRef, useState } from "react";

import { splitToolLine } from "../../lib/text";
import type { TraceLine, TraceVoice } from "../../state/pipeline";
import { PixelBars, PixelIcon, PixelSpinner, Typewriter } from "../Pixel";

const VOICE_CLASS = { model: "n-model", code: "n-code", human: "n-human" } as const;
const VOICES: TraceVoice[] = ["model", "code", "human"];

/**
 * The LCD: the agent trace. Each step has a pixel glyph; lines that arrive
 * while you watch type themselves out; tool calls show the call and, under
 * it, what code answered.
 */
export function TraceScreen({ label, lines, live }: { label: string; lines: TraceLine[]; live: boolean }) {
  const end = useRef<HTMLLIElement>(null);
  // Lines present on first render are history; only later ones animate.
  // The parent keys this component by run, so switching runs starts fresh.
  const [lastSeen] = useState(() => lines.at(-1)?.seq ?? -1);
  const fresh = (seq: number) => seq > lastSeen;

  useEffect(() => {
    if (live) end.current?.scrollIntoView({ block: "nearest" });
  }, [lines.length, live]);

  const counts = VOICES.map((v) => [v, lines.filter((l) => l.voice === v).length] as const);

  return (
    <div className="lcd flex min-h-40 flex-1 flex-col" role="log" aria-label={`Agent trace for ${label}`} tabIndex={0}>
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[#a3a39c]">
        <PixelBars live={live} />
        <span>trace / {label}</span>
        <span className="ml-auto flex items-center gap-3">
          {counts.map(([v, n]) => (
            <span key={v} className="inline-flex items-center gap-1.5" title={`${n} ${v} lines`}>
              <i className={`px-sq ${VOICE_CLASS[v]}`} aria-hidden="true" />
              {n}
            </span>
          ))}
          <span className={live ? "n-human" : undefined}>{live ? "streaming" : "idle"}</span>
        </span>
      </div>
      <ol className="m-0 flex min-h-0 flex-1 list-none flex-col gap-1.5 overflow-y-auto p-0">
        {lines.map((l, i) => {
          const tool = l.node === "tool" ? splitToolLine(l.text) : null;
          const isNew = fresh(l.seq);
          return (
            <li key={l.seq} className={`trace-row ${isNew ? "trace-new" : ""} ${l.node === "guardrail" ? "trace-guard" : ""}`}>
              <span className="text-[#8e8e87]">{String(i + 1).padStart(2, "0")}</span>
              <span className={`flex items-center gap-2 ${VOICE_CLASS[l.voice]}`}>
                <PixelIcon node={l.node} className="px-pop" />
                {l.node}
              </span>
              {tool ? (
                <span className="flex min-w-0 flex-col">
                  <span className="text-[#f0f0ea]">
                    <Typewriter text={tool.call} animate={isNew} />
                  </span>
                  {tool.result && (
                    <span className="flex gap-2 text-[#a3a39c]">
                      <span aria-hidden="true">↳</span>
                      <span className="min-w-0">
                        <Typewriter text={tool.result} animate={isNew} />
                      </span>
                    </span>
                  )}
                </span>
              ) : (
                <span className="min-w-0">
                  <Typewriter text={l.text} animate={isNew} />
                </span>
              )}
            </li>
          );
        })}
        {live && (
          <li ref={end} className="trace-row n-human">
            <span />
            <span className="flex items-center gap-2">
              <PixelSpinner />
              running
            </span>
            <span className="px-caret" aria-hidden="true" />
          </li>
        )}
        {lines.length === 0 && !live && <li className="text-[#a3a39c]">No trace yet.</li>}
      </ol>
      <div className="mt-3 flex flex-wrap gap-x-4 text-[#a3a39c]">
        <span className="n-model">blue: model</span>
        <span className="n-code">green: code</span>
        <span className="n-human">orange: human or live</span>
      </div>
    </div>
  );
}
