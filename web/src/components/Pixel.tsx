import { useEffect, useState } from "react";

import { spriteFor } from "../lib/sprites";

/** A 7x7 pixel glyph for a trace step, drawn crisp at any size. */
export function PixelIcon({ node, size = 14, className = "" }: { node: string; size?: number; className?: string }) {
  const rows = spriteFor(node);
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 7 7"
      shapeRendering="crispEdges"
      aria-hidden="true"
      className={`flex-none ${className}`}
    >
      {rows.flatMap((row, y) =>
        [...row].map((c, x) => (c === "#" ? <rect key={`${x}-${y}`} x={x} y={y} width={1} height={1} fill="currentColor" /> : null)),
      )}
    </svg>
  );
}

const SPIN = ["▖", "▘", "▝", "▗"];

/** A four-frame block spinner for the step that is running. */
export function PixelSpinner() {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    if (reducedMotion()) return;
    const id = setInterval(() => setFrame((f) => (f + 1) % SPIN.length), 120);
    return () => clearInterval(id);
  }, []);
  return (
    <span className="n-human" aria-hidden="true">
      {SPIN[frame]}
    </span>
  );
}

/**
 * Text that types itself out once, for lines that arrive while you watch.
 * Lines already there on load, and viewers who prefer reduced motion, see it whole.
 */
export function Typewriter({ text, animate }: { text: string; animate: boolean }) {
  const [shown, setShown] = useState(() => (animate && !reducedMotion() ? 0 : text.length));
  const typing = shown < text.length;
  useEffect(() => {
    if (!typing) return;
    // About 400ms for a line, whatever its length.
    const step = Math.max(2, Math.ceil(text.length / 24));
    const id = setInterval(() => setShown((n) => Math.min(text.length, n + step)), 16);
    return () => clearInterval(id);
  }, [typing, text.length]);
  return (
    <>
      {text.slice(0, shown)}
      {typing && <span className="px-caret" aria-hidden="true" />}
    </>
  );
}

function reducedMotion(): boolean {
  try {
    return matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}
