import { useCallback, useEffect, useMemo, useRef } from 'react';

/** OS-drawn circle. Chromium allows 128px; our slider tops out at 80. */
function cssBrushCursor(size, eraser) {
  const s = Math.max(6, Math.round(Number(size) || 24));
  const canvas = document.createElement('canvas');
  canvas.width = s;
  canvas.height = s;
  const ctx = canvas.getContext('2d');
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
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
    radius: Math.max(1, brushCss * (canvas.width / rect.width) / 2),
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

export function maskPngFromCanvas(canvas) {
  const src = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
  const out = document.createElement('canvas');
  out.width = canvas.width;
  out.height = canvas.height;
  const dst = out.getContext('2d').createImageData(canvas.width, canvas.height);
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
  out.getContext('2d').putImageData(dst, 0, 0);
  return out.toDataURL('image/png');
}

export default function InpaintBrushEditor({
  src,
  alt,
  disabled = false,
  eraser = false,
  brushCss = 24,
  onDirty,
  onNaturalSize,
  canvasRef,
}) {
  const imageRef = useRef(null);
  const localCanvasRef = useRef(null);
  const dragRef = useRef(null);
  const onDirtyRef = useRef(onDirty);
  const onNaturalSizeRef = useRef(onNaturalSize);
  onDirtyRef.current = onDirty;
  onNaturalSizeRef.current = onNaturalSize;
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
    onNaturalSizeRef.current?.(image.naturalWidth, image.naturalHeight);
  }, []);

  useEffect(() => {
    const canvas = localCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    onDirtyRef.current?.(false);
    fitCanvas();
  }, [src, fitCanvas]);

  const paint = useCallback((event) => {
    const canvas = localCanvasRef.current;
    if (!canvas || disabled) return;
    const point = canvasPoint(canvas, event, brushCss);
    if (!point) return;
    const ctx = canvas.getContext('2d');
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
    <div
      className="relative inline-block max-h-[100cqh] max-w-[100cqw] leading-none"
      role="group"
      aria-label="Touch-up brush"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
    >
      <img
        ref={imageRef}
        src={src}
        alt={alt}
        draggable={false}
        onLoad={fitCanvas}
        onDragStart={(event) => event.preventDefault()}
        className="block max-h-[100cqh] max-w-[100cqw] select-none"
      />
      <canvas
        ref={setCanvas}
        aria-label="Paint the area to inpaint"
        className="absolute inset-0 h-full w-full touch-none"
        style={{ cursor: disabled ? 'default' : cursorCss, opacity: 0.7 }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onLostPointerCapture={endDrag}
      />
    </div>
  );
}
