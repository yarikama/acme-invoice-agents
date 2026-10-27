export function money(amount: number | null | undefined, currency = "USD"): string {
  if (amount == null) return "n/a";
  try {
    return amount.toLocaleString("en-US", { style: "currency", currency });
  } catch {
    // An unknown currency code from a messy invoice.
    return `${currency} ${amount.toFixed(2)}`;
  }
}

export function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}
