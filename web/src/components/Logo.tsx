/** Signal sheet: an invoice whose corner is a signal light. */
export function Logo({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="32 24 192 208" aria-hidden="true">
      <path
        fill="currentColor"
        d="M80 32 H128 A16 16 0 0 1 144 48 V72 A16 16 0 0 0 160 88 H184 A16 16 0 0 1 200 104 V200 A24 24 0 0 1 176 224 H80 A24 24 0 0 1 56 200 V56 A24 24 0 0 1 80 32 Z"
      />
      <circle fill="#FF5A1F" cx="180" cy="52" r="20" />
    </svg>
  );
}
