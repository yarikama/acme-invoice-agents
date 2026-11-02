import { ArrowCounterClockwiseIcon, PlayIcon, StackIcon, UploadSimpleIcon } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";

import { api, type Parser } from "../api/client";
import type { RunOverview, Sample } from "../api/types";

interface Props {
  onStarted: (runs: RunOverview[]) => void;
}

// What each parser does, for its option and tooltip.
const PARSERS: { value: Parser; label: string; hint: string }[] = [
  { value: "pymupdf", label: "PyMuPDF", hint: "Tesseract reads the whole page; fast, free" },
  { value: "docling", label: "Docling", hint: "A layout model finds tables first; Markdown" },
  { value: "grok", label: "Grok vision", hint: "Grok reads the page image; best on poor scans" },
];

export function Toolbar({ onStarted }: Props) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sample, setSample] = useState("");
  // Which parser reads scans in the runs started here; the API's default first.
  const [parser, setParser] = useState<Parser>("pymupdf");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Starting over forgets every run, so it asks once more, in the page.
  const [confirming, setConfirming] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.health().then(
      (h) => h.parser && setParser(h.parser),
      () => {},
    );
    api.samples().then(
      (list) => {
        setSamples(list);
        setSample((current) => current || (list[0]?.name ?? ""));
      },
      () => setError("Can't reach the API. Start it with make api, then reload."),
    );
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

  // The event stream tells every open console to clear its list.
  const startOver = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.reset();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-end gap-2">
        <button type="button" className="key key-sm" disabled={busy} onClick={() => run(() => api.startBatch(parser))}>
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
          onClick={() => run(() => api.startSample(sample, parser))}
        >
          <PlayIcon size={16} aria-hidden="true" />
          Run
        </button>
        <label className="flex flex-col gap-1" title={PARSERS.find((p) => p.value === parser)?.hint}>
          <span className="cap">scan parser</span>
          <select
            className="key key-sm appearance-none bg-raised pr-8"
            value={parser}
            onChange={(e) => setParser(e.target.value as Parser)}
          >
            {PARSERS.map((p) => (
              <option key={p.value} value={p.value} title={p.hint}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="key key-sm" disabled={busy} onClick={() => fileInput.current?.click()}>
          <UploadSimpleIcon size={16} aria-hidden="true" />
          Upload invoice
        </button>
        <input
          ref={fileInput}
          type="file"
          className="sr-only"
          accept=".txt,.csv,.json,.xml,.pdf,.png,.jpg,.jpeg"
          aria-label="Invoice file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) run(() => api.upload(file, parser));
          }}
        />
        {confirming ? (
          <span className="flex items-center gap-2" role="group" aria-label="Start over">
            <span className="text-sm">Forget every run and invoice received?</span>
            <button type="button" className="key key-sm" disabled={busy} onClick={startOver}>
              Start over
            </button>
            <button type="button" className="key key-sm" onClick={() => setConfirming(false)}>
              Keep them
            </button>
          </span>
        ) : (
          <button
            type="button"
            className="key key-sm"
            disabled={busy}
            title="Forget past runs, so the samples aren't flagged as duplicates"
            onClick={() => setConfirming(true)}
          >
            <ArrowCounterClockwiseIcon size={16} aria-hidden="true" />
            Start over
          </button>
        )}
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
