import type { Icon } from "@phosphor-icons/react";

/** A category icon in a small rounded box; `signal` marks what needs a person. */
export function Badge({ icon: Glyph, signal = false }: { icon: Icon; signal?: boolean }) {
  return (
    <span className={`badge ${signal ? "badge-signal" : ""}`} aria-hidden="true">
      <Glyph size={18} weight="bold" />
    </span>
  );
}
