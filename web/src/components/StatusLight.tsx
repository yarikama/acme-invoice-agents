import type { Status } from "../api/types";
import { STATUS_LABEL } from "../state/runs";

/** A status light with its word: shape and fill differ, never colour alone. */
export function StatusLight({ status, withLabel = true }: { status: Status; withLabel?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap">
      <span className={`lt lt-${status}`} aria-hidden="true" />
      {withLabel && <span>{STATUS_LABEL[status]}</span>}
    </span>
  );
}
