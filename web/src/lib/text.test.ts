import { describe, expect, it } from "vitest";

import { sentences, splitToolLine } from "./text";

describe("sentences", () => {
  it("splits prose but keeps decimals, money and abbreviations in brackets whole", () => {
    expect(
      sentences(
        "Converted EUR 4,125.00 to $4,455.00 at 1.08, which is under the threshold. WidgetA at 225 EUR (~$243) is fine. (Checked twice.) Done!",
      ),
    ).toEqual([
      "Converted EUR 4,125.00 to $4,455.00 at 1.08, which is under the threshold.",
      "WidgetA at 225 EUR (~$243) is fine.",
      "(Checked twice.)",
      "Done!",
    ]);
  });

  it("returns nothing for empty text", () => {
    expect(sentences("  ")).toEqual([]);
  });
});

describe("splitToolLine", () => {
  it("separates the call from what code answered", () => {
    expect(splitToolLine("check_inventory(GadgetX): GadgetX: 5 in stock, list price $750.00.")).toEqual({
      call: "check_inventory(GadgetX)",
      result: "GadgetX: 5 in stock, list price $750.00.",
    });
  });

  it("keeps a line without a result as the call", () => {
    expect(splitToolLine("vendor_history(Acme)")).toEqual({ call: "vendor_history(Acme)", result: "" });
  });
});
