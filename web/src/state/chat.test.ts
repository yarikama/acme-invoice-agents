import { describe, expect, it } from "vitest";

import type { ChatMessage } from "../api/types";
import { lastIndex, merge } from "./chat";

const msg = (index: number, text = `m${index}`): ChatMessage => ({
  index,
  role: index % 2 ? "assistant" : "user",
  text,
  via: "web",
  at: null,
  commands: [],
});

describe("chat", () => {
  it("merges a reply and a poll that overlap, in order", () => {
    const got = merge([msg(0), msg(1)], [msg(3), msg(2), msg(1, "again")]);
    expect(got.map((m) => [m.index, m.text])).toEqual([
      [0, "m0"],
      [1, "again"],
      [2, "m2"],
      [3, "m3"],
    ]);
    expect(merge(got, [])).toBe(got);
    expect(lastIndex(got)).toBe(3);
    expect(lastIndex([])).toBe(-1);
  });
});
