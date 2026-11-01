import { PlayIcon, StackIcon, UploadSimpleIcon } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { RunOverview, Sample } from "../api/types";

interface Props {
  onStarted: (runs: RunOverview[]) => void;
}

export function Toolbar({ onStarted }: Props) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sample, setSample] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.samples().then((list) => {
      setSamples(list);
      setSample((current) => current || (list[0]?.name ?? ""));
    }, () => setError("Can't reach the API. Start it with make api, then reload."));
  }, []);

  const run = async (start: () => Promise<RunOverview | RunOverview[]>) => {
    setBusy(true);
    setError(null);
    try {
      const result = await start();
      onStarted(Array.isArray(result) ? result : [result]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-end gap-2">
        <button type="button" className="key key-sm" disabled={busy} onClick={() => run(api.startBatch)}>
          <StackIcon size={16} aria-hidden="true" />
          Run all samples
        </button>
        <label className="flex flex-col gap-1">
          <span className="cap">sample</span>
          <select
            className="key key-sm min-w-56 appearance-none bg-raised pr-8"
            value={sample}
            onChange={(e) => setSample(e.target.value)}
          >
            {samples.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="key key-sm"
          disabled={busy || !sample}
          onClick={() => run(() => api.startSample(sample))}
        >
          <PlayIcon size={16} aria-hidden="true" />
          Run
        </button>
        <button type="button" className="key key-sm" disabled={busy} onClick={() => fileInput.current?.click()}>
          <UploadSimpleIcon size={16} aria-hidden="true" />
          Upload invoice
        </button>
        <input
          ref={fileInput}
          type="file"
          className="sr-only"
          accept=".txt,.csv,.json,.xml,.pdf"
          aria-label="Invoice file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) run(() => api.upload(file));
          }}
        />
      </div>
      {error && (
        <p role="alert" className="m-0 text-sm">
          <span className="tag mr-2">error</span>
          {error}
        </p>
      )}
    </div>
  );
}
