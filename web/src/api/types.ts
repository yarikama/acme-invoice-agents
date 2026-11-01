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

export interface Payment {
  status: string;
  vendor: string;
  amount: number;
  currency: string;
  // The bank's idempotency key, and whether it was already paid before.
  idempotency_key?: string | null;
  duplicate?: boolean;
}

/** What a step produced, on its trace line: the extraction, the checks, the payment. */
export interface StageOutput {
  invoice?: Invoice;
  corrections?: string[];
  findings?: Finding[];
  payment?: Payment;
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
  // Why the rules send this invoice to a VP, when they do.
  route_reason?: string | null;
  draft: { decision: Decision; rationale: string } | null;
  payment: Payment | null;
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

/** Tokens and cost of the model calls one step made. */
export interface StepUsage {
  // The model that served the step, as the provider names it.
  model?: string;
  input_tokens: number;
  output_tokens: number;
  // Null when the model's price is unknown.
  cost_usd: number | null;
  // Answered from a recording (replay): tokens and cost are the live run's.
  replayed?: boolean;
}

interface EventBase {
  id: number;
  seq: number;
  run_id: string;
  source: string;
  created_at: string;
  // When the event happened (ISO 8601); older servers may leave it out.
  at?: string;
  status: Status;
}

export type RunEvent = EventBase &
  (
    | { type: "started" }
    | { type: "progress"; stage: string; step: string }
    // `tool` is set when the line records a tool call the approver made.
    | {
        type: "trace";
        node: string;
        stage: string;
        message: string;
        tool?: string;
        usage?: StepUsage;
        data?: StageOutput;
      }
    | { type: "awaiting_vp"; request: unknown; summary: Summary }
    | { type: "decision"; approved: boolean; note: string }
    | { type: "done"; summary: Summary }
    | { type: "run_error"; message: string }
    // Someone started over: every run and received invoice was forgotten.
    | { type: "reset" }
  );

/** One message of the assistant's conversation, shared with the terminal. */
export interface ChatMessage {
  index: number;
  role: "user" | "assistant";
  text: string;
  // Where it was said: "terminal" or "web".
  via: string | null;
  at: string | null;
  // The console commands the assistant ran for it.
  commands: string[];
}

export interface ChatState {
  messages: ChatMessage[];
  // A live model is configured, so the assistant can answer.
  available: boolean;
  // Redis holds the conversation, so the terminal sees it too.
  shared: boolean;
}

export interface Sample {
  name: string;
  format: string;
  bytes: number;
}
