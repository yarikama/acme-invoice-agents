import { MoonIcon, SunIcon } from "@phosphor-icons/react";
import { useState } from "react";

import type { StreamState } from "../state/useConsole";
import { Logo } from "./Logo";

const STREAM_WORD: Record<StreamState, string> = {
  connecting: "connecting",
  live: "live",
  retrying: "reconnecting",
};

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

export function Header({ stream, provider }: { stream: StreamState; provider: string | null }) {
  const { dark, toggle } = useTheme();
  return (
    <header className="border-b border-seam bg-panel">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center gap-x-6 gap-y-2 px-6 py-3">
        <div className="flex items-center gap-2">
          <Logo size={22} />
          <span className="text-base font-medium">Acme AP</span>
          <span className="cap">console</span>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-4">
          <span className="inline-flex items-center gap-2" aria-live="polite">
            <span className={`lt ${stream === "live" ? "lt-paid" : "lt-running"}`} aria-hidden="true" />
            <span className="cap">stream</span>
            <span className="text-sm">{STREAM_WORD[stream]}</span>
          </span>
          <span className="tag" title="LLM provider configured on the server">
            {provider ?? "server offline"}
          </span>
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
