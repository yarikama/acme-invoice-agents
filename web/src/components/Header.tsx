import { ArrowCounterClockwiseIcon, MoonIcon, SunIcon } from "@phosphor-icons/react";
import { useState } from "react";

import type { StreamState } from "../state/useConsole";
import { Logo } from "./Logo";


function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const toggle = () => {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      // Private mode: the choice just isn't remembered.
    }
    setDark(next);
  };
  return { dark, toggle };
}

export function Header({ stream, onResetLayout }: { stream: StreamState; onResetLayout: () => void }) {
  const { dark, toggle } = useTheme();
  return (
    <header className="bar-ink sticky top-0 z-10">
      <div className="flex w-full flex-wrap items-center gap-x-6 gap-y-2 px-4 py-4 sm:px-6">
        <div className="flex items-center gap-3">
          <Logo size={40} />
          <span className="display text-[28px] leading-none">Acme AP</span>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-4">
          {/* Silent while connected; speaks up only when updates have stopped arriving. */}
          <span aria-live="polite">
            {stream === "retrying" && (
              <span className="inline-flex items-center gap-2 text-sm">
                <span className="lt lt-running" aria-hidden="true" />
                Reconnecting…
              </span>
            )}
          </span>
          {/* Only the wide grid can be rearranged, so only it can be reset. */}
          <button
            type="button"
            className="key key-sm hidden min-[900px]:inline-flex"
            onClick={onResetLayout}
            title="Put every widget back where it started"
          >
            <ArrowCounterClockwiseIcon size={16} />
            Reset layout
          </button>
          <button
            type="button"
            className="key key-sm"
            onClick={toggle}
            aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {dark ? <SunIcon size={16} /> : <MoonIcon size={16} />}
          </button>
        </div>
      </div>
    </header>
  );
}
