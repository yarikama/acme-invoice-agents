import type { ChatMessage } from "../api/types";

/** Messages merged by index: a poll and a reply may both carry one. */
export function merge(current: ChatMessage[], incoming: ChatMessage[]): ChatMessage[] {
  if (incoming.length === 0) return current;
  const byIndex = new Map(current.map((m) => [m.index, m]));
  for (const m of incoming) byIndex.set(m.index, m);
  return [...byIndex.values()].sort((a, b) => a.index - b.index);
}

/** The last index we hold, for asking only what came after. */
export function lastIndex(messages: ChatMessage[]): number {
  return messages.length ? messages[messages.length - 1]!.index : -1;
}
