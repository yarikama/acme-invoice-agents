/**
 * Split prose into sentences for a list. Breaks after . ! ? (and any closing
 * bracket or quote) followed by a space and a capital or an opening bracket, so "1.08" and "$4,455.00" stay whole.
 */
export function sentences(text: string): string[] {
  return text
    .split(/(?<=[.!?][)"']?)\s+(?=[A-Z(~$€£"'])/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** A tool line "name(args): result" as its call and its result. */
export function splitToolLine(text: string): { call: string; result: string } {
  const at = text.indexOf("): ");
  if (at === -1) return { call: text, result: "" };
  return { call: text.slice(0, at + 1), result: text.slice(at + 3) };
}
