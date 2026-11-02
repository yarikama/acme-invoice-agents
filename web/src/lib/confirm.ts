// Paying at least this much takes a second click.
const CONFIRM_ABOVE = 10_000;

/** Whether approving takes a second click: $10,000 or more, or any other
 * currency, since the policy's threshold is in US dollars. */
export function needsSecondClick(total: number | null, currency: string | null): boolean {
  return (currency ?? "USD").toUpperCase() !== "USD" || (total ?? 0) >= CONFIRM_ABOVE;
}
