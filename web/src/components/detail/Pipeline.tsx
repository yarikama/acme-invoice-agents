import type { Segment } from "../../state/pipeline";

const LED_CLASS: Record<Segment["state"], string> = {
  off: "led-off",
  run: "led-run",
  done: "led-done",
  flag: "led-flag",
  block: "led-fail",
  fail: "led-fail",
  wait: "led-wait",
  skip: "led-skip",
};

export function Pipeline({ segments }: { segments: Segment[] }) {
  return (
    <ol aria-label="Pipeline stages" className="m-0 grid list-none grid-cols-3 gap-2 p-0 sm:grid-cols-6">
      {segments.map((s) => (
        <li key={s.key} className="flex min-w-0 flex-col gap-1">
          <span className={`led ${LED_CLASS[s.state]}`} aria-hidden="true" />
          <span className="cap mt-1 text-ink">{s.label}</span>
          <span className="text-[13px] font-semibold">{s.word}</span>
        </li>
      ))}
    </ol>
  );
}
