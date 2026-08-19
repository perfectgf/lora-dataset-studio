/**
 * 🖌 The repair BRUSH — paint the pixels to replace instead of boxing them.
 *
 * Contributed by OneCodingDude on GitHub (PR #37). A rectangle is the wrong shape
 * for most of what people actually want gone: a necklace, a pair of glasses, a
 * strap. Boxing one of those hands the model a square full of face it was never
 * asked to touch, and the reconstruction drifts accordingly.
 *
 * The canvas is kept at the image's NATURAL size, not its displayed size, so the
 * mask that leaves here lines up with the file on disk whatever the screen did
 * with the picture. `maskPngFromCanvas` then hardens it: anything the brush
 * touched becomes pure white, everything else pure black, because a soft alpha
 * edge would reach the server as "repaint this a bit", which no lane means.
 *
 * Pointer events (not mouse) with capture, and `touch-none`: this has to work
 * under a finger, and a phone that scrolls the page while you paint is not a
 * brush.
 *
 * SIZING — the wrapper and the img carry the SAME viewport-based lengths, and
 * the canvas overlays with `inset-0`. Both halves of that matter:
 *
 *   - `max-h-full` on the img was a no-op. It resolves against the wrapper's
 *     content height, which is itself defined by the img — indefinite, so the
 *     used value is `none`. Nothing bounded the height: the picture grew to the
 *     full stage width, overflowed downwards and covered the prompt/Repair row
 *     underneath it. (The contribution used container-query units; they were
 *     swapped for `max-h-full` on the way in, which is where this came from.
 *     They are back, with a `vh` floor for hosts that declare no container.)
 *   - because the wrapper is `inline-block` and shrink-wraps a single block img
 *     with no padding, wrapper box == img box, so `absolute inset-0` puts the
 *     canvas exactly on the displayed picture. When the img was allowed to
 *     outgrow the wrapper, the canvas kept the WRAPPER's smaller box: a click
 *     39.5% down the visible image landed at 50% in the mask, and the bottom of
 *     the picture could not be reached at all. Same construct the box editor
 *     (WatermarkRegionEditor) has always used, for the same reason.
 *
 * Any change here must keep `img.getBoundingClientRect()` equal to the canvas's.
 */
import { useCallback, useEffect, useMemo, useRef } from 'react';

/** OS-drawn circle so the cursor IS the brush size. Chromium caps at 128 px. */
function cssBrushCursor(size, eraser) {
  const s = Math.max(6, Math.min(120, Math.round(Number(size) || 24)));
  const canvas = document.createElement('canvas');
  canvas.width = s;
  canvas.height = s;
  const ctx = canvas.getContext('2d');
  if (!ctx) return 'crosshair';
  const c = s / 2;
  const r = Math.max(1, c - 1.5);
  ctx.beginPath();
  ctx.arc(c, c, r, 0, Math.PI * 2);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(c, c, r, 0, Math.PI * 2);
  ctx.strokeStyle = eraser ? '#111' : '#f472b6';
  ctx.lineWidth = 1;
  ctx.stroke();
  const hot = Math.floor(c);
  return `url(${canvas.toDataURL('image/png')}) ${hot} ${hot}, crosshair`;
}

function canvasPoint(canvas, event, brushCss) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const scale = canvas.width / rect.width;
  return {
    x: (event.clientX - rect.left) * scale,
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
    radius: Math.max(1, (brushCss * scale) / 2),
  };
}

function stroke(ctx, from, to, erase) {
  ctx.save();
  ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over';
  ctx.strokeStyle = 'rgb(244, 114, 182)';
  ctx.fillStyle = ctx.strokeStyle;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = to.radius * 2;
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(to.x, to.y, to.radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/** The painted canvas as a hard black/white PNG data URL — null if untouched. */
export function maskPngFromCanvas(canvas) {
  if (!canvas || !canvas.width || !canvas.height) return null;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const src = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const out = document.createElement('canvas');
  out.width = canvas.width;
  out.height = canvas.height;
  const outCtx = out.getContext('2d');
  const dst = outCtx.createImageData(canvas.width, canvas.height);
  let painted = false;
  for (let i = 0; i < src.data.length; i += 4) {
    const on = src.data[i + 3] > 8;
    const v = on ? 255 : 0;
    if (on) painted = true;
    dst.data[i] = v;
    dst.data[i + 1] = v;
    dst.data[i + 2] = v;
    dst.data[i + 3] = 255;
  }
  if (!painted) return null;
  outCtx.putImageData(dst, 0, 0);
  return out.toDataURL('image/png');
}

export default function InpaintBrushEditor({
  src,
  alt = 'image',
  disabled = false,
  eraser = false,
  brushCss = 28,
  onDirty,
  canvasRef,
}) {
  const imageRef = useRef(null);
  const localCanvasRef = useRef(null);
  const dragRef = useRef(null);
  const onDirtyRef = useRef(onDirty);
  onDirtyRef.current = onDirty;
  const cursorCss = useMemo(() => cssBrushCursor(brushCss, eraser), [brushCss, eraser]);

  const setCanvas = (node) => {
    localCanvasRef.current = node;
    if (typeof canvasRef === 'function') canvasRef(node);
    else if (canvasRef) canvasRef.current = node;
  };

  const fitCanvas = useCallback(() => {
    const image = imageRef.current;
    const canvas = localCanvasRef.current;
    if (!image || !canvas || !image.naturalWidth) return;
    if (canvas.width !== image.naturalWidth || canvas.height !== image.naturalHeight) {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
    }
  }, []);

  // A new image (or a fresh repair on the same one) starts from a blank mask.
  useEffect(() => {
    const canvas = localCanvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    onDirtyRef.current?.(false);
    fitCanvas();
  }, [src, fitCanvas]);

  const paint = useCallback((event) => {
    const canvas = localCanvasRef.current;
    if (!canvas || disabled) return;
    const point = canvasPoint(canvas, event, brushCss);
    if (!point) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const drag = dragRef.current;
    stroke(ctx, drag?.last || point, point, eraser);
    dragRef.current = { pointerId: event.pointerId, last: point, captureTarget: event.currentTarget };
    onDirtyRef.current?.(true);
  }, [brushCss, disabled, eraser]);

  const onPointerDown = useCallback((event) => {
    if (disabled || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    paint(event);
  }, [disabled, paint]);

  const onPointerMove = useCallback((event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    paint(event);
  }, [paint]);

  const endDrag = useCallback((event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    try {
      if (drag.captureTarget?.hasPointerCapture?.(event.pointerId)) {
        drag.captureTarget.releasePointerCapture(event.pointerId);
      }
    } catch { /* capture already gone */ }
  }, []);

  return (
    <div className="relative inline-block max-h-[min(70vh,calc(100cqh_-_1.5rem))] max-w-[min(92vw,100cqw)] leading-none"
      role="group" aria-label="Repair brush"
      onClick={(event) => event.stopPropagation()}>
      {/* No object-fit class here: the element box already carries the
          intrinsic ratio, and a fitted box that differs from the element box is
          precisely the drift the canvas cannot see. */}
      <img ref={imageRef} src={src} alt={alt} draggable={false}
        onLoad={fitCanvas}
        onDragStart={(event) => event.preventDefault()}
        className="block max-h-[min(70vh,calc(100cqh_-_1.5rem))] max-w-[min(92vw,100cqw)] select-none" />
      <canvas ref={setCanvas}
        aria-label="Paint the area to repair"
        className="absolute inset-0 h-full w-full touch-none"
        style={{ cursor: disabled ? 'default' : cursorCss, opacity: 0.65 }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onLostPointerCapture={endDrag} />
    </div>
  );
}
