import { useRef, useCallback } from 'react';

/* Drag-by-header for a board-space card: pointer capture on the header,
   deltas divided by the board scale so a drag follows the cursor at any
   zoom. Presses on buttons inside the header never arm a drag. */
export function useBoardDrag(boardScale, onMove, onCommit) {
  const drag = useRef(null);
  const onPointerDown = useCallback((e) => {
    if (e.button != null && e.button !== 0) return;
    if (e.target.closest('button')) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY };
  }, []);
  const onPointerMove = useCallback((e) => {
    if (!drag.current) return;
    const k = boardScale || 1;
    onMove((e.clientX - drag.current.x) / k, (e.clientY - drag.current.y) / k);
    drag.current = { x: e.clientX, y: e.clientY };
  }, [boardScale, onMove]);
  const end = useCallback(() => {
    if (!drag.current) return;
    drag.current = null;
    onCommit?.();
  }, [onCommit]);
  return { onPointerDown, onPointerMove, onPointerUp: end, onPointerCancel: end };
}
