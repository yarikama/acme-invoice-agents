import type { LayoutItem } from "react-grid-layout";

// The console's widgets, in the order they stack on narrow screens.
export const WIDGETS = ["overview", "inbox", "summary", "decision", "stream", "findings", "lines"] as const;
export type WidgetId = (typeof WIDGETS)[number];

export const COLS = 12;
export const ROW_HEIGHT = 40;
export const GAP = 16;

const MIN: Record<WidgetId, { minW: number; minH: number }> = {
  overview: { minW: 6, minH: 4 },
  inbox: { minW: 4, minH: 5 },
  summary: { minW: 3, minH: 5 },
  decision: { minW: 3, minH: 6 },
  stream: { minW: 4, minH: 7 },
  findings: { minW: 3, minH: 3 },
  lines: { minW: 3, minH: 4 },
};

// Bento default: the inbox and the trace on the left, the selected invoice's
// facts on the right, the decision right under its summary.
export const DEFAULT_LAYOUT: LayoutItem[] = (
  [
    { i: "overview", x: 0, y: 0, w: 12, h: 4 },
    { i: "inbox", x: 0, y: 4, w: 7, h: 7 },
    { i: "summary", x: 7, y: 4, w: 5, h: 5 },
    { i: "decision", x: 7, y: 9, w: 5, h: 13 },
    { i: "stream", x: 0, y: 11, w: 7, h: 24 },
    { i: "findings", x: 7, y: 22, w: 5, h: 6 },
    { i: "lines", x: 7, y: 28, w: 5, h: 7 },
  ] as const
).map((item) => ({ ...item, ...MIN[item.i] }));

// Bump when the default changes, so browsers holding the old default get the new one.
const KEY = "acme-console-layout-v4";

function isItem(v: unknown): v is LayoutItem {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return (
    typeof o.i === "string" &&
    (WIDGETS as readonly string[]).includes(o.i) &&
    ["x", "y", "w", "h"].every((k) => Number.isInteger(o[k]) && (o[k] as number) >= 0) &&
    (o.w as number) > 0 &&
    (o.h as number) > 0 &&
    (o.x as number) + (o.w as number) <= COLS
  );
}

/**
 * A saved layout, cleaned up: unknown or malformed items are dropped, and any
 * widget the save doesn't mention (e.g. one added later) gets its default spot.
 */
export function restore(saved: unknown): LayoutItem[] {
  const items = Array.isArray(saved) ? saved.filter(isItem) : [];
  const byId = new Map(items.map((it) => [it.i, it]));
  return DEFAULT_LAYOUT.map((d) => {
    const s = byId.get(d.i);
    return s ? { ...d, x: s.x, y: s.y, w: Math.max(s.w, d.minW ?? 1), h: Math.max(s.h, d.minH ?? 1) } : d;
  });
}

export function loadLayout(storage: Pick<Storage, "getItem"> | null = safeStorage()): LayoutItem[] {
  try {
    const raw = storage?.getItem(KEY);
    return raw ? restore(JSON.parse(raw)) : DEFAULT_LAYOUT;
  } catch {
    return DEFAULT_LAYOUT;
  }
}

export function saveLayout(layout: readonly LayoutItem[], storage: Pick<Storage, "setItem"> | null = safeStorage()) {
  try {
    storage?.setItem(KEY, JSON.stringify(layout.map(({ i, x, y, w, h }) => ({ i, x, y, w, h }))));
  } catch {
    // Private mode or full storage: the layout just isn't remembered.
  }
}

export function clearLayout(storage: Pick<Storage, "removeItem"> | null = safeStorage()) {
  try {
    storage?.removeItem(KEY);
  } catch {
    // Nothing to clear.
  }
}

function safeStorage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
