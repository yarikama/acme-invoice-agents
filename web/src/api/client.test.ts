import { afterEach, describe, expect, it, vi } from "vitest";

import { api, ApiError } from "./client";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(status: number, body: unknown) {
  const fetch = vi.fn(async () => new Response(JSON.stringify(body), { status }));
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

describe("api.decide", () => {
  it("sends a VP token as a bearer token, and nothing without one", async () => {
    const fetch = stubFetch(202, { id: "r1" });
    await api.decide("r1", true, "ok", "s3cret");
    await api.decide("r1", false, "no");
    const [withToken, without] = fetch.mock.calls.map((c) => (c as unknown as [string, RequestInit])[1]);
    expect(withToken!.headers).toMatchObject({ Authorization: "Bearer s3cret", "Content-Type": "application/json" });
    expect(without!.headers).not.toHaveProperty("Authorization");
    expect(JSON.parse(String(without!.body))).toEqual({ approved: false, note: "no" });
  });

  it("surfaces a refused token as a 401", async () => {
    stubFetch(401, { detail: "Only a VP can approve or reject: send a VP token." });
    await expect(api.decide("r1", true, "")).rejects.toMatchObject({ status: 401 });
    await expect(api.decide("r1", true, "")).rejects.toBeInstanceOf(ApiError);
  });
});

describe("the scan parser", () => {
  it("goes with every way of starting a run, and is left out when unset", async () => {
    const fetch = stubFetch(202, []);
    await api.startSample("invoice_1001.txt", "grok");
    await api.startBatch("docling");
    await api.upload(new File(["x"], "scan.png"), "pymupdf");
    await api.upload(new File(["x"], "scan.png"));
    const inits = fetch.mock.calls.map((c) => (c as unknown as [string, RequestInit])[1]);
    expect(JSON.parse(String(inits[0]!.body))).toEqual({ sample: "invoice_1001.txt", parser: "grok" });
    expect(JSON.parse(String(inits[1]!.body))).toEqual({ parser: "docling" });
    expect((inits[2]!.body as FormData).get("parser")).toBe("pymupdf");
    expect((inits[3]!.body as FormData).get("parser")).toBeNull();
  });
});
