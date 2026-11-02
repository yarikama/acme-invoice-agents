import { useCallback, useEffect, useReducer, useState } from "react";

import { api } from "../api/client";
import type { RunEvent, RunOverview } from "../api/types";
import { initialRuns, runsReducer } from "./runs";

// Not "error": that name is EventSource's own connection-error event.
const EVENT_TYPES = ["started", "progress", "trace", "awaiting_vp", "decision", "done", "run_error", "reset"];

export type StreamState = "connecting" | "live" | "retrying";

// How long to wait before opening a new stream after the old one gave up.
const RETRY_MS = 3000;

/** All runs, kept current by the one SSE stream at /api/events. */
export function useConsole() {
  const [state, dispatch] = useReducer(runsReducer, initialRuns);
  const [stream, setStream] = useState<StreamState>("connecting");
  const [provider, setProvider] = useState<string | null>(null);
  // No API key was set, so the agents' answers come from a recording.
  const [noKey, setNoKey] = useState(false);
  // The API takes VP decisions only with a token.
  const [vpAuth, setVpAuth] = useState(false);
  // False until the first run list arrives, so the page doesn't claim to be empty.
  const [loaded, setLoaded] = useState(false);

  const snapshot = useCallback((runs: RunOverview[]) => dispatch({ type: "snapshot", runs }), []);

  useEffect(() => {
    api.health().then(
      (h) => {
        setProvider(h.llm_provider);
        setNoKey(Boolean(h.no_key));
        setVpAuth(Boolean(h.vp_auth));
      },
      () => setProvider(null),
    );
    api
      .runs()
      .then(snapshot, () => {})
      .finally(() => setLoaded(true));

    // EventSource reconnects by itself and sends Last-Event-ID, so the
    // server replays only what was missed. It gives up for good on an error
    // response (a dev proxy's 502 while the API is down), so then this opens
    // a new one after a pause.
    let source: EventSource | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let closed = false;
    let boot: string | undefined;
    const onEvent = (msg: MessageEvent<string>) => {
      const event = JSON.parse(msg.data) as RunEvent;
      // A new API process: the reducer starts over from its log, and the run
      // list brings back the runs it restored (paused for a VP).
      if (event.boot && boot && event.boot !== boot) api.runs().then(snapshot, () => {});
      boot = event.boot ?? boot;
      dispatch({ type: "event", event });
    };
    const connect = () => {
      source = new EventSource("/api/events");
      for (const type of EVENT_TYPES) source.addEventListener(type, onEvent);
      source.onopen = () => setStream("live");
      source.onerror = () => {
        setStream("retrying");
        if (source?.readyState === EventSource.CLOSED && !closed) {
          retry = setTimeout(connect, RETRY_MS);
        }
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      source?.close();
    };
  }, [snapshot]);

  return { state, stream, provider, noKey, vpAuth, snapshot, loaded };
}
