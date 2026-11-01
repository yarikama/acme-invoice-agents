import { describe, expect, it } from "vitest";

import { spreadLabels } from "./labels";

describe("spreadLabels", () => {
  it("leaves labels that don't touch where they are", () => {
    expect(spreadLabels([20, 100, 180], 18, 0, 200)).toEqual([20, 100, 180]);
  });

  it("stacks labels on the same value instead of overlapping them", () => {
    // Paid at 1, "Waiting for a VP" and "Rejected" both at 0 (the bottom).
    const [paid, vp, rejected] = spreadLabels([110, 212, 212], 18, 10, 212);
    expect(paid).toBe(110);
    expect(rejected).toBe(212);
    expect(vp).toBe(194); // pushed up one line, keeping the series order
  });

  it("pushes a crowded stack back up off the bottom edge", () => {
    const ys = spreadLabels([200, 200, 200], 18, 10, 200);
    expect(ys).toEqual([164, 182, 200]);
  });

  it("keeps every label inside the bounds and at least a gap apart", () => {
    const ys = spreadLabels([5, 6, 7, 300], 18, 10, 120);
    const sorted = [...ys].sort((a, b) => a - b);
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(10);
    expect(Math.max(...ys)).toBeLessThanOrEqual(120);
    for (let k = 1; k < sorted.length; k++) {
      expect(sorted[k]! - sorted[k - 1]!).toBeGreaterThanOrEqual(18);
    }
  });
});
