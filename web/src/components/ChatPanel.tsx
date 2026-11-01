import { PaperPlaneRightIcon } from "@phosphor-icons/react";
import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { ChatMessage } from "../api/types";
import { lastIndex, merge } from "../state/chat";
import { PixelSpinner } from "./Pixel";

// How often to look for messages typed in the terminal.
const POLL_MS = 2000;

function Message({ m }: { m: ChatMessage }) {
  const mine = m.role === "user";
  return (
    <li className={`flex flex-col gap-1 ${mine ? "items-end" : "items-start"}`}>
      <span className="text-xs text-ink-2">
        {mine ? "Request" : "Assistant"}
        {m.via && <span className="tag ml-2">{m.via}</span>}
      </span>
      <div className={`bubble max-w-[85%] text-sm ${mine ? "bg-raised" : ""}`}>
        <p className="m-0 whitespace-pre-wrap">{m.text}</p>
        {m.commands.length > 0 && (
          <ul className="m-0 mt-2 flex list-none flex-col gap-1 p-0">
            {m.commands.map((c, i) => (
              <li key={i} className="mono text-xs text-[var(--code)]">
                → {c}
              </li>
            ))}
          </ul>
        )}
      </div>
    </li>
  );
}

/**
 * The assistant, in the console: one conversation with the terminal's. What
 * is typed in either shows in both, and is context for the next request.
 */
export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [available, setAvailable] = useState(true);
  const [shared, setShared] = useState(true);
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const end = useRef<HTMLDivElement>(null);
  // The messages for the poll, which runs outside render.
  const held = useRef<ChatMessage[]>([]);
  useEffect(() => {
    held.current = messages;
  }, [messages]);

  const poll = useCallback(async () => {
    try {
      const got = await api.chat(lastIndex(held.current));
      setAvailable(got.available);
      setShared(got.shared);
      setMessages((now) => merge(now, got.messages));
    } catch {
      // The API is down or restarting; the next poll tries again.
    }
  }, []);

  useEffect(() => {
    void poll();
    const timer = setInterval(() => void poll(), POLL_MS);
    return () => clearInterval(timer);
  }, [poll]);

  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" });
  }, [messages.length, sending]);

  const send = async (e: FormEvent) => {
    e.preventDefault();
    const request = text.trim();
    if (!request || sending) return;
    setSending(true);
    setError(null);
    try {
      const got = await api.say(request);
      setMessages((now) => merge(now, got.messages));
      setText("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <ol className="m-0 flex min-h-0 flex-1 list-none flex-col gap-3 overflow-y-auto p-0 pr-3" aria-live="polite">
        {messages.length === 0 && (
          <li className="text-sm text-ink-2">
            Ask in plain words: "run invoice 1004", "what's waiting for a VP?", "why was 1009 rejected?". The terminal
            shares this conversation.
          </li>
        )}
        {messages.map((m) => (
          <Message key={m.index} m={m} />
        ))}
        {sending && (
          <li className="flex items-center gap-2 text-sm">
            <span className="avatar avatar-model avatar-active" aria-hidden="true">
              <PixelSpinner />
            </span>
            <span className="shimmer">Thinking…</span>
          </li>
        )}
        <div ref={end} />
      </ol>
      {!shared && (
        <p className="m-0 text-xs text-ink-2">
          Not shared with the terminal: no Redis at REDIS_URL, so each side keeps its own conversation.
        </p>
      )}
      <form className="flex flex-none gap-2" onSubmit={send}>
        <input
          className="key min-h-10 flex-1 px-3 text-sm shadow-none active:translate-y-0"
          placeholder={available ? "Ask the assistant…" : "The assistant needs a live model (LLM_PROVIDER)"}
          aria-label="Message to the assistant"
          value={text}
          disabled={!available || sending}
          maxLength={2000}
          onChange={(e) => setText(e.target.value)}
        />
        <button type="submit" className="key key-sm" disabled={!available || sending || !text.trim()}>
          <PaperPlaneRightIcon size={16} aria-hidden="true" />
          Send
        </button>
      </form>
      {error && (
        <p role="alert" className="m-0 text-sm">
          <span className="tag mr-2">error</span>
          {error}
        </p>
      )}
    </div>
  );
}
