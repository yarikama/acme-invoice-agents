/**
 * Vertical positions for direct labels that would otherwise overlap: each
 * label as close to its mark as it can be, at least `gap` apart, inside
 * [min, max]. Two lines ending on the same value get one label above the
 * other instead of two on top of each other.
 *
 * Returns the positions in the order given.
 */
export function spreadLabels(wanted: number[], gap: number, min: number, max: number): number[] {
  const order = wanted.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y || a.i - b.i);
  const placed = order.map((o) => Math.min(Math.max(o.y, min), max));
  // Down the stack: each label at least `gap` below the one above it...
  for (let k = 1; k < placed.length; k++) placed[k] = Math.max(placed[k]!, placed[k - 1]! + gap);
  // ...then back up from the bottom edge if that pushed any past `max`.
  for (let k = placed.length - 1; k >= 0; k--) {
    const below = k === placed.length - 1 ? max : placed[k + 1]! - gap;
    placed[k] = Math.min(placed[k]!, below);
  }
  const out = new Array<number>(wanted.length);
  order.forEach((o, k) => (out[o.i] = placed[k]!));
  return out;
}
