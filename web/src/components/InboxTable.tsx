import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useMemo } from "react";

import { money } from "../lib/format";
import type { RunView } from "../state/runs";
import { StatusLight } from "./StatusLight";

const col = createColumnHelper<RunView>();

export function InboxTable({
  runs,
  loaded,
  filtered,
  selectedId,
  onSelect,
}: {
  runs: RunView[];
  loaded: boolean;
  // True when a status filter hid every run; the page header covers "no runs at all".
  filtered: boolean;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const columns = useMemo(
    () => [
      col.display({
        id: "invoice",
        header: "invoice",
        cell: ({ row }) => {
          const inv = row.original.summary?.invoice;
          return (
            <button
              type="button"
              className="flex min-h-12 w-full cursor-pointer flex-col justify-center border-0 bg-transparent p-0 text-left text-ink hover:[&>.v]:underline"
              aria-current={row.original.id === selectedId}
              onClick={() => onSelect(row.original.id)}
            >
              <span className="mono text-xs text-ink-2">{inv?.invoice_number ?? row.original.source}</span>
              <span className="v text-base font-medium underline-offset-[3px]">
                {inv?.vendor ?? (inv ? "No vendor" : row.original.status === "failed" ? "Not extracted" : "Not extracted yet")}
              </span>
            </button>
          );
        },
      }),
      col.display({
        id: "format",
        header: "format",
        cell: ({ row }) => (
          <span className="tag">{(row.original.summary?.source_format ?? row.original.source.split(".").pop() ?? "").toUpperCase()}</span>
        ),
      }),
      col.display({
        id: "amount",
        header: () => <span className="block text-right">amount</span>,
        cell: ({ row }) => {
          const inv = row.original.summary?.invoice;
          return <span className="block text-right whitespace-nowrap">{inv ? money(inv.total, inv.currency) : ""}</span>;
        },
      }),
      col.display({
        id: "status",
        header: "status",
        cell: ({ row }) => <StatusLight status={row.original.status} />,
      }),
      col.display({
        id: "flags",
        header: () => <span className="block text-right">flags</span>,
        cell: ({ row }) => {
          const flags = (row.original.summary?.findings ?? []).filter((f) => f.severity !== "info").length;
          return <span className="block text-right text-ink-2">{row.original.summary ? flags : ""}</span>;
        },
      }),
    ],
    [selectedId, onSelect],
  );

  // TanStack Table v8 returns unstable functions; fine without the React Compiler.
  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({ data: runs, columns, getCoreRowModel: getCoreRowModel(), getRowId: (r) => r.id });

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse">
        <thead>
          {table.getHeaderGroups().map((group) => (
            <tr key={group.id}>
              {group.headers.map((header) => (
                <th
                  key={header.id}
                  scope="col"
                  className="mono px-4 py-2 text-left text-xs font-normal text-ink-2 first:pl-6 last:pr-6"
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className={`group ${row.id === selectedId ? "bg-raised" : "odd:bg-[var(--tile)] hover:bg-raised"}`}>
              {row.getVisibleCells().map((cell, i) => (
                <td
                  key={cell.id}
                  className={`relative px-4 py-1.5 align-middle text-sm first:pl-6 last:pr-6 ${
                    i === 0 && row.id === selectedId
                      ? "before:absolute before:top-2 before:bottom-2 before:left-2 before:w-1 before:rounded-full before:bg-ink"
                      : ""
                  }`}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {runs.length === 0 && !loaded && (
        // Placeholder rows the shape of real ones while the run list loads.
        <div aria-busy="true" aria-label="Loading invoices" className="flex flex-col">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex items-center gap-6 px-6 py-4">
              <span className="flex flex-1 flex-col gap-2">
                <span className="h-3 w-20 rounded bg-pale" />
                <span className="h-3 w-40 rounded bg-pale" />
              </span>
              <span className="h-5 w-12 rounded bg-pale" />
              <span className="h-3 w-20 rounded bg-pale" />
            </div>
          ))}
        </div>
      )}
      {runs.length === 0 && loaded && filtered && (
        <p className="m-0 px-6 py-8 text-ink-2">No invoices with this status.</p>
      )}
    </div>
  );
}
