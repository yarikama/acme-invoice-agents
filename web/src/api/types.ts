// Mirrors the FastAPI responses in src/invoice_agents/api.

export type Status = "queued" | "running" | "awaiting_vp" | "paid" | "rejected" | "failed";
export type Severity = "error" | "warning" | "info";
export type Decision = "approve" | "reject" | "escalate";

export interface LineItem {
  item: string;
  quantity: number;
  unit_price: number;
  amount: number | null;
  note: string | null;
}

export interface Invoice {
  invoice_number: string | null;
  revision: string | null;
  vendor: string | null;
  vendor_address: string | null;
  invoice_date: string | null;
  due_date: string | null;
  currency: string;
  line_items: LineItem[];
  subtotal: number | null;
  tax_amount: number | null;
  shipping: number | null;
  total: number | null;
  payment_terms: string | null;
  notes: string | null;
}

export interface Finding {
  code: string;
  severity: Severity;
  title: string;
  detail: string;
}

export interface TraceEntry {
  stage: string;
  message: string;
}

export interface Summary {
  status: Status;
  source_format: string | null;
  invoice: Invoice | null;
  findings: Finding[];
  route: "reject" | "human" | "auto" | null;
  draft: { decision: Decision; rationale: string } | null;
  payment: { status: string; vendor: string; amount: number; currency: string } | null;
  reason: string | null;
  trace: TraceEntry[];
}

export interface RunOverview {
  id: string;
  source: string;
  status: Status;
  created_at: string;
  invoice_number: string | null;
  vendor: string | null;
  total: number | null;
  currency: string | null;
}

interface EventBase {
  id: number;
  seq: number;
  run_id: string;
  source: string;
  created_at: string;
  status: Status;
}

export type RunEvent = EventBase &
  (
    | { type: "started" }
    | { type: "progress"; stage: string; step: string }
    | { type: "trace"; node: string; stage: string; message: string }
    | { type: "awaiting_vp"; request: unknown; summary: Summary }
    | { type: "decision"; approved: boolean; note: string }
    | { type: "done"; summary: Summary }
    | { type: "run_error"; message: string }
  );

export interface Sample {
  name: string;
  format: string;
  bytes: number;
}
