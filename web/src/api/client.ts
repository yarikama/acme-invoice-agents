import type { RunOverview, Sample } from "./types";

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
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<{ status: string; llm_provider: string }>("/api/health"),
  samples: () => request<Sample[]>("/api/samples"),
  runs: () => request<RunOverview[]>("/api/runs"),
  startSample: (sample: string) => request<RunOverview>("/api/runs", json({ sample })),
  startBatch: () => request<RunOverview[]>("/api/runs/batch", { method: "POST" }),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<RunOverview>("/api/runs/upload", { method: "POST", body: form });
  },
  decide: (runId: string, approved: boolean, note: string) =>
    request<RunOverview>(`/api/runs/${runId}/decision`, json({ approved, note })),
};
