import type { ChatMessage, ChatState, RunOverview, Sample } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, body?.detail ?? res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/** What reads a scanned PDF or photo. */
export type Parser = "pymupdf" | "docling" | "grok";

export const api = {
  health: () =>
    request<{
      status: string;
      llm_provider: string;
      vp_auth: boolean;
      no_key?: boolean;
      parser?: Parser;
      parsers?: Parser[];
      boot?: string;
    }>("/api/health"),
  samples: () => request<Sample[]>("/api/samples"),
  runs: () => request<RunOverview[]>("/api/runs"),
  startSample: (sample: string, parser?: Parser) => request<RunOverview>("/api/runs", json({ sample, parser })),
  startBatch: (parser?: Parser) => request<RunOverview[]>("/api/runs/batch", json({ parser })),
  upload: (file: File, parser?: Parser) => {
    const form = new FormData();
    form.append("file", file);
    if (parser) form.append("parser", parser);
    return request<RunOverview>("/api/runs/upload", { method: "POST", body: form });
  },
  /** Forget every run and received invoice, so the samples can run again. */
  reset: (token?: string) =>
    request<void>("/api/reset", {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    }),
  /** The assistant's conversation after message `after`. */
  chat: (after = -1) => request<ChatState>(`/api/chat?after=${after}`),
  /** Ask the assistant; returns the messages the request added. */
  say: (text: string) => request<{ messages: ChatMessage[] }>("/api/chat", json({ text })),
  /** The VP's answer; `token` is a VP's bearer token when the API requires one. */
  decide: (runId: string, approved: boolean, note: string, token?: string) => {
    const init = json({ approved, note });
    if (token) init.headers = { ...init.headers, Authorization: `Bearer ${token}` };
    return request<RunOverview>(`/api/runs/${runId}/decision`, init);
  },
};
