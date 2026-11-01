import { useEffect, useRef, useState } from "react";

/** The width of an element, kept current as it resizes (widgets resize). */
export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry!.contentRect.width)));
    ro.observe(node);
    return () => ro.disconnect();
  }, []);
  return { ref, width };
}
