import { describe, expect, it } from "vitest";

import { clearLayout, COLS, DEFAULT_LAYOUT, loadLayout, restore, saveLayout, WIDGETS } from "./layout";

function memory() {
  const data = new Map<string, string>();
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
  };
}

describe("layout", () => {
  it("places every widget in the default, inside the grid, without overlaps", () => {
    expect(DEFAULT_LAYOUT.map((l) => l.i)).toEqual([...WIDGETS]);
    for (const a of DEFAULT_LAYOUT) {
      expect(a.x + a.w).toBeLessThanOrEqual(COLS);
      for (const b of DEFAULT_LAYOUT) {
        if (a === b) continue;
        const apart = a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y;
        expect(apart, `${a.i} overlaps ${b.i}`).toBe(true);
      }
    }
  });

  it("round-trips a saved layout", () => {
    const store = memory();
    const moved = DEFAULT_LAYOUT.map((l) => (l.i === "inbox" ? { ...l, x: 5, w: 7 } : l));
    saveLayout(moved, store);
    expect(loadLayout(store).find((l) => l.i === "inbox")).toMatchObject({ x: 5, w: 7 });
    clearLayout(store);
    expect(loadLayout(store)).toEqual(DEFAULT_LAYOUT);
  });

  it("drops junk, fills in missing widgets and keeps minimum sizes", () => {
    const got = restore([
      { i: "inbox", x: 0, y: 0, w: 1, h: 1 }, // below its minimum
      { i: "ghost", x: 0, y: 0, w: 2, h: 2 }, // not a widget
      { i: "stream", x: 10, y: 0, w: 6, h: 8 }, // off the grid
      "nonsense",
    ]);
    expect(got.map((l) => l.i)).toEqual([...WIDGETS]);
    const inbox = got.find((l) => l.i === "inbox")!;
    expect(inbox.w).toBe(inbox.minW);
    expect(inbox.h).toBe(inbox.minH);
    expect(got.find((l) => l.i === "stream")).toEqual(DEFAULT_LAYOUT.find((l) => l.i === "stream"));
    expect(restore({ not: "an array" })).toEqual(DEFAULT_LAYOUT);
  });

  it("falls back to the default when storage is broken", () => {
    expect(loadLayout({ getItem: () => "{not json" })).toEqual(DEFAULT_LAYOUT);
    expect(
      loadLayout({
        getItem: () => {
          throw new Error("denied");
        },
      }),
    ).toEqual(DEFAULT_LAYOUT);
    expect(() =>
      saveLayout(DEFAULT_LAYOUT, {
        setItem: () => {
          throw new Error("quota");
        },
      }),
    ).not.toThrow();
    expect(loadLayout(null)).toEqual(DEFAULT_LAYOUT);
  });
});
