import { DotsSixVerticalIcon, type Icon } from "@phosphor-icons/react";
import type { ReactNode } from "react";

import { Badge } from "./Badge";

/**
 * A dashboard card. Its title bar is the drag handle; the body scrolls when
 * the card is smaller than its content.
 */
export function Widget({
  id,
  title,
  icon,
  aside,
  signal = false,
  flush = false,
  draggable = true,
  children,
}: {
  id: string;
  title: string;
  icon: Icon;
  aside?: ReactNode;
  // Orange badge and edge: this card needs a person.
  signal?: boolean;
  // Body without padding, for full-bleed tables.
  flush?: boolean;
  draggable?: boolean;
  children: ReactNode;
}) {
  return (
    <section
      aria-labelledby={`${id}-h`}
      className={`card flex h-full min-h-0 flex-col overflow-hidden ${signal ? "card-signal" : ""}`}
    >
      <header className={`flex items-center gap-3 px-5 pt-4 pb-3 ${draggable ? "widget-handle" : ""}`}>
        <Badge icon={icon} signal={signal} />
        <h2 id={`${id}-h`} className="m-0 min-w-0 truncate text-lg font-medium">
          {title}
        </h2>
        <span className="ml-auto flex min-w-0 items-center gap-3">
          {aside}
          {draggable && (
            <DotsSixVerticalIcon size={18} weight="bold" className="flex-none text-ink-2" aria-hidden="true" />
          )}
        </span>
      </header>
      <div className={`min-h-0 flex-1 overflow-auto ${flush ? "" : "px-5 pb-5"}`}>{children}</div>
    </section>
  );
}
