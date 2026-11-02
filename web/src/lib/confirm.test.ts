import { describe, expect, it } from "vitest";

import { needsSecondClick } from "./confirm";

describe("the second click to pay", () => {
  it("is asked at $10,000 or more, and for any other currency", () => {
    expect(needsSecondClick(9_999, "USD")).toBe(false);
    expect(needsSecondClick(10_000, "USD")).toBe(true);
    // EUR 9,500 is over the USD threshold; the threshold is in dollars.
    expect(needsSecondClick(9_500, "EUR")).toBe(true);
    expect(needsSecondClick(9_500, "usd")).toBe(false);
    expect(needsSecondClick(null, null)).toBe(false);
  });
});
