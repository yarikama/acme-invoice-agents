import { useCallback, useEffect, useReducer, useState } from "react";

import { api } from "../api/client";
import type { RunEvent, RunOverview } from "../api/types";
import { initialRuns, runsReducer } from "./runs";

// Not "error": that name is EventSource's own connection-error event.
const EVENT_TYPES = ["started", "progress", "trace", "awaiting_vp", "decision", "done", "run_error"];

export type StreamState = "connecting" | "live" | "retrying";

/** All runs, kept current by the one SSE stream at /api/events. */
export function useConsole() {
  const [state, dispatch] = useReducer(runsReducer, initialRuns);
  const [stream, setStream] = useState<StreamState>("connecting");
  const [provider, setProvider] = useState<string | null>(null);
  // False until the first run list arrives, so the page doesn't claim to be empty.
  const [loaded, setLoaded] = useState(false);

  const snapshot = useCallback((runs: RunOverview[]) => dispatch({ type: "snapshot", runs }), []);

  useEffect(() => {
    api.health().then((h) => setProvider(h.llm_provider), () => setProvider(null));
    api
      .runs()
      .then(snapshot, () => {})
      .finally(() => setLoaded(true));

    // EventSource reconnects by itself and sends Last-Event-ID, so the
    // server replays only what was missed.
    const source = new EventSource("/api/events");
    const onEvent = (msg: MessageEvent<string>) => {
      dispatch({ type: "event", event: JSON.parse(msg.data) as RunEvent });
    };
    for (const type of EVENT_TYPES) source.addEventListener(type, onEvent);
    source.onopen = () => setStream("live");
    source.onerror = () => setStream("retrying");
    return () => source.close();
  }, [snapshot]);

  return { state, stream, provider, snapshot, loaded };
}
