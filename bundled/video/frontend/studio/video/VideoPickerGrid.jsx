import { useLayoutEffect, useRef, useState } from 'react';
import { gridBoxHeight } from './videoPickerTile';

const GAP = 4;
const OVERSCAN = 2;

/** Keep only visible rows (and two nearby rows) mounted, regardless of how
 * many pages were loaded. Selection belongs to the picker, never these tiles. */
export default function VideoPickerGrid({ count, tile, label, itemKey, children }) {
  const box = useRef(null);
  const pendingFocus = useRef(null);
  const [viewport, setViewport] = useState({ width: 0, height: 0, top: 0 });
  const [focused, setFocused] = useState(null);

  useLayoutEffect(() => {
    const element = box.current;
    const measure = () => setViewport((previous) => {
      const next = { width: element.clientWidth, height: element.clientHeight, top: element.scrollTop };
      return Object.keys(next).every((key) => previous[key] === next[key]) ? previous : next;
    });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    element.addEventListener('scroll', measure, { passive: true });
    return () => { observer.disconnect(); element.removeEventListener('scroll', measure); };
  }, []);

  const columns = Math.max(1, Math.floor((viewport.width + GAP) / (tile + GAP)));
  const side = Math.max(1, (viewport.width - GAP * (columns - 1)) / columns);
  const stride = side + GAP;
  const rows = Math.ceil(count / columns);
  const height = Math.max(0, rows * stride - GAP);
  // Clamp immediately after resizing or replacing a long list with a short one.
  const top = Math.min(viewport.top, Math.max(0, height - viewport.height));
  const first = Math.max(0, Math.floor(top / stride) - OVERSCAN) * columns;
  const end = Math.min(count, (Math.ceil((top + viewport.height) / stride) + OVERSCAN) * columns);
  const indices = Array.from({ length: end - first }, (_, index) => first + index);
  // A mouse scroll must not remove the focused button from the document.
  if (focused !== null && focused < count && (focused < first || focused >= end)) {
    indices.push(focused);
    indices.sort((a, b) => a - b);
  }

  useLayoutEffect(() => {
    if (pendingFocus.current === null) return;
    const button = box.current.querySelector(`[data-picker-index="${pendingFocus.current}"] button`);
    if (button) { button.focus({ preventScroll: true }); pendingFocus.current = null; }
  });

  const keyDown = (event) => {
    const cell = event.target.closest('[data-picker-index]');
    if (!cell || event.altKey || event.ctrlKey || event.metaKey) return;
    const index = Number(cell.dataset.pickerIndex);
    const page = Math.max(1, Math.floor(viewport.height / stride)) * columns;
    const moves = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: columns, ArrowUp: -columns,
      PageDown: page, PageUp: -page, Tab: event.shiftKey ? -1 : 1 };
    let next;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = count - 1;
    else if (event.key in moves) next = index + moves[event.key];
    else return;
    // Tab leaves the gallery normally at either end.
    if (event.key === 'Tab' && (next < 0 || next >= count)) return;
    event.preventDefault();
    next = Math.max(0, Math.min(count - 1, next));
    const y = Math.floor(next / columns) * stride;
    const element = box.current;
    if (y < element.scrollTop) element.scrollTop = y;
    else if (y + side > element.scrollTop + element.clientHeight) element.scrollTop = y + side - element.clientHeight;
    pendingFocus.current = next;
    setFocused(next);
    setViewport((value) => ({ ...value, top: element.scrollTop }));
  };

  return (
    <div ref={box} role="group" aria-label={label} className="relative overflow-y-auto"
      style={{ height: `min(${height}px, ${gridBoxHeight(tile)}px, 70vh)`, scrollbarGutter: 'stable' }}
      onKeyDown={keyDown}
      onFocusCapture={(event) => {
        const cell = event.target.closest('[data-picker-index]');
        if (cell) setFocused(Number(cell.dataset.pickerIndex));
      }}
      onBlurCapture={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setFocused(null); }}>
      <div className="relative" style={{ height }}>
        {indices.map((index) => (
          <div key={itemKey(index)} data-picker-index={index} className="absolute"
            style={{ top: Math.floor(index / columns) * stride, left: (index % columns) * stride,
              width: side, height: side }}>
            {children(index)}
          </div>
        ))}
      </div>
    </div>
  );
}
