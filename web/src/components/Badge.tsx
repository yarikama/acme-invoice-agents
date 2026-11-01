import type { Icon } from "@phosphor-icons/react";

/** A category icon in a small rounded box; `signal` marks what needs a person. */
export function Badge({ icon: Glyph, signal = false }: { icon: Icon; signal?: boolean }) {
  return (
    <span className={`badge ${signal ? "badge-signal" : ""}`} aria-hidden="true">
      <Glyph size={16} weight="bold" />
    </span>
  );
}

/** A section heading with its icon. */
export function Heading({ id, icon, children }: { id: string; icon: Icon; children: string }) {
  return (
    <h3 id={id} className="m-0 flex items-center gap-3 text-base font-medium">
      <Badge icon={icon} />
      {children}
    </h3>
  );
}
