/** Manual crop editor with a STRETCHABLE box: drag the 8 handles to resize, drag
 * inside to move — any shape, not just square (training buckets non-square
 * images fine). Ratio presets snap the box; `defaultAspect` presets the initial
 * ratio (the reference crop starts at 1:1 — its historical convention — but
 * stays freely reshapeable: nothing downstream actually requires a square).
 * `lockSquare` pins 1:1 and hides the ratio row (kept for callers that need it).
 * Returns the box in NATURAL image pixels, same contract as before.
 * Custom implementation (no react-easy-crop): that lib pins a fixed frame and
 * moves the image under it — it cannot stretch the selection itself. */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useI18n } from '../../i18n/I18nContext';

const ASPECTS = [
  ['free', null],
  ['1:1', 1],
  ['3:4', 3 / 4],
  ['2:3', 2 / 3],
  ['9:16', 9 / 16],
  ['4:3', 4 / 3],
  ['3:2', 3 / 2],
  ['16:9', 16 / 9],
];
const MIN_SIDE = 32;   // natural px — below this a crop is useless for training

// Handles: name -> which box edges the drag moves.
const HANDLES = [
  ['nw', { left: true, top: true }], ['n', { top: true }], ['ne', { right: true, top: true }],
  ['w', { left: true }], ['e', { right: true }],
  ['sw', { left: true, bottom: true }], ['s', { bottom: true }], ['se', { right: true, bottom: true }],
];
const HANDLE_POS = {
  nw: 'left-0 top-0 -translate-x-1/2 -translate-y-1/2 cursor-nwse-resize',
  n: 'left-1/2 top-0 -translate-x-1/2 -translate-y-1/2 cursor-ns-resize',
  ne: 'right-0 top-0 translate-x-1/2 -translate-y-1/2 cursor-nesw-resize',
  w: 'left-0 top-1/2 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize',
  e: 'right-0 top-1/2 translate-x-1/2 -translate-y-1/2 cursor-ew-resize',
  sw: 'left-0 bottom-0 -translate-x-1/2 translate-y-1/2 cursor-nesw-resize',
  s: 'left-1/2 bottom-0 -translate-x-1/2 translate-y-1/2 cursor-ns-resize',
  se: 'right-0 bottom-0 translate-x-1/2 translate-y-1/2 cursor-nwse-resize',
};

function clampBox(b, W, H) {
  const w = Math.min(Math.max(b.w, MIN_SIDE), W);
  const h = Math.min(Math.max(b.h, MIN_SIDE), H);
  return { x: Math.min(Math.max(b.x, 0), W - w), y: Math.min(Math.max(b.y, 0), H - h), w, h };
}

// Best centered box of `ratio` (w/h) fitting inside W x H, centered on the current box.
function ratioBox(cur, ratio, W, H) {
  let w = cur.w; let h = w / ratio;
  if (h > H) { h = H; w = h * ratio; }
  if (w > W) { w = W; h = w / ratio; }
  const cx = cur.x + cur.w / 2; const cy = cur.y + cur.h / 2;
  return clampBox({ x: cx - w / 2, y: cy - h / 2, w, h }, W, H);
}

export default function CropModal({ imageUrl, onCancel, onConfirm, onReset,
                                    lockSquare = false, defaultAspect = null }) {
  const { t } = useI18n();
  const [nat, setNat] = useState(null);        // {W, H} natural size
  const [box, setBox] = useState(null);        // crop box in NATURAL px
  const [aspect, setAspect] = useState(lockSquare ? 1 : defaultAspect);   // null = free
  const imgRef = useRef(null);
  const cancelRef = useRef(null);
  const dragRef = useRef(null);                // {mode, start, startBox}

  const onImgLoad = (e) => {
    const W = e.target.naturalWidth; const H = e.target.naturalHeight;
    setNat({ W, H });
    // Initial box: largest centered box of the initial ratio, or 80% of the frame.
    const a = lockSquare ? 1 : defaultAspect;
    if (a) {
      let w = W; let h = w / a;
      if (h > H) { h = H; w = h * a; }
      setBox({ x: (W - w) / 2, y: (H - h) / 2, w, h });
    } else {
      setBox(clampBox({ x: W * 0.1, y: H * 0.1, w: W * 0.8, h: H * 0.8 }, W, H));
    }
  };

  // Escape closes; initial focus on Cancel.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onCancel(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel]);
  useEffect(() => { cancelRef.current?.focus(); }, []);
  // The overlay maps natural px -> displayed px via the live img rect: re-render
  // on window resize so it stays glued to the image.
  const [, forceRender] = useState(0);
  useEffect(() => {
    const onResize = () => forceRender((n) => n + 1);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const scale = () => {
    const el = imgRef.current;
    return el && nat ? el.getBoundingClientRect().width / nat.W : 1;
  };

  const startDrag = (e, mode) => {
    e.preventDefault(); e.stopPropagation();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    dragRef.current = { mode, sx: e.clientX, sy: e.clientY, start: { ...box } };
  };

  const onMove = useCallback((e) => {
    const d = dragRef.current;
    if (!d || !nat) return;
    const s = scale();
    const dx = (e.clientX - d.sx) / s;
    const dy = (e.clientY - d.sy) / s;
    const { W, H } = nat;
    const b0 = d.start;
    if (d.mode === 'move') {
      setBox(clampBox({ ...b0, x: b0.x + dx, y: b0.y + dy }, W, H));
      return;
    }
    const edges = Object.fromEntries(HANDLES)[d.mode] || {};
    let x1 = b0.x, y1 = b0.y, x2 = b0.x + b0.w, y2 = b0.y + b0.h;
    if (edges.left) x1 = Math.min(Math.max(0, x1 + dx), x2 - MIN_SIDE);
    if (edges.right) x2 = Math.max(Math.min(W, x2 + dx), x1 + MIN_SIDE);
    if (edges.top) y1 = Math.min(Math.max(0, y1 + dy), y2 - MIN_SIDE);
    if (edges.bottom) y2 = Math.max(Math.min(H, y2 + dy), y1 + MIN_SIDE);
    let nb = { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
    const ratio = lockSquare ? 1 : aspect;
    if (ratio) {
      // Constrain to the ratio: the dominant dragged dimension wins, the other
      // follows; anchor on the opposite corner/edge so the grab point tracks.
      const horiz = !!(edges.left || edges.right);
      const vert = !!(edges.top || edges.bottom);
      if (horiz && !vert) nb.h = nb.w / ratio;
      else if (vert && !horiz) nb.w = nb.h * ratio;
      else if (Math.abs(dx) >= Math.abs(dy)) nb.h = nb.w / ratio;
      else nb.w = nb.h * ratio;
      if (edges.left) nb.x = x2 - nb.w;
      if (edges.top) nb.y = y2 - nb.h;
      // keep inside the frame — shrink if the ratio pushed it out
      if (nb.x < 0) { nb.w += nb.x; nb.h = nb.w / ratio; nb.x = 0; if (edges.top) nb.y = y2 - nb.h; }
      if (nb.y < 0) { nb.h += nb.y; nb.w = nb.h * ratio; nb.y = 0; if (edges.left) nb.x = x2 - nb.w; }
      if (nb.x + nb.w > W) { nb.w = W - nb.x; nb.h = nb.w / ratio; if (edges.top) nb.y = y2 - nb.h; }
      if (nb.y + nb.h > H) { nb.h = H - nb.y; nb.w = nb.h * ratio; if (edges.left) nb.x = x2 - nb.w; }
    }
    setBox(clampBox(nb, W, H));
  }, [nat, aspect, lockSquare]);

  const endDrag = useCallback(() => { dragRef.current = null; }, []);
  useEffect(() => {
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', endDrag);
    return () => { window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', endDrag); };
  }, [onMove, endDrag]);

  const pickAspect = (value) => {
    setAspect(value);
    if (value && box && nat) setBox(ratioBox(box, value, nat.W, nat.H));
  };

  const s = scale();
  return (
    <div role="dialog" aria-modal="true" aria-label={t('workspace.crop.title')}
      className="fixed inset-0 z-[9995] bg-black/85 flex flex-col p-3 sm:p-4">
      <div className="relative flex-1 min-h-0 w-full max-w-4xl mx-auto flex items-center justify-center overflow-hidden">
        <div className="relative inline-block max-h-full max-w-full">
          <img ref={imgRef} src={imageUrl} alt={t('workspace.crop.imageAlt')} onLoad={onImgLoad} draggable={false}
            className="max-h-[70vh] max-w-full object-contain select-none block" />
          {box && nat && (
            <div
              className="absolute border-2 border-indigo-400 cursor-move touch-none"
              style={{ left: box.x * s, top: box.y * s, width: box.w * s, height: box.h * s,
                       boxShadow: '0 0 0 9999px rgba(0,0,0,0.55)' }}
              onPointerDown={(e) => startDrag(e, 'move')}
              role="application" aria-label={t('workspace.crop.selectionLabel')}>
              {HANDLES.map(([name]) => (
                <span key={name}
                  onPointerDown={(e) => startDrag(e, name)}
                  className={`absolute w-3.5 h-3.5 rounded-full bg-indigo-400 border-2 border-white/90 touch-none ${HANDLE_POS[name]}`} />
              ))}
              <span className="absolute -top-6 left-0 px-1.5 py-0.5 rounded bg-black/70 text-white text-[10px] tabular-nums pointer-events-none">
                {Math.round(box.w)}×{Math.round(box.h)}
              </span>
            </div>
          )}
        </div>
      </div>
      <div className="shrink-0 w-full max-w-4xl mx-auto mt-3 flex flex-col gap-2">
        {!lockSquare && (
          <div className="flex items-center gap-1.5 flex-wrap" role="group" aria-label={t('workspace.crop.aspectLabel')}>
            <span className="text-white/60 text-xs">{t('workspace.crop.ratio')}</span>
            {ASPECTS.map(([label, value]) => (
              <button key={label} type="button" onClick={() => pickAspect(value)}
                aria-pressed={aspect === value}
                className={`px-2 py-0.5 rounded text-xs font-semibold ${aspect === value
                  ? 'bg-indigo-500 text-white'
                  : 'bg-white/10 text-white/70 hover:bg-white/20'}`}>
                {label}
              </button>
            ))}
            <span className="text-white/40 text-[10px]">{t('workspace.crop.freeHint')}</span>
          </div>
        )}
        <div className="flex gap-2 justify-end">
          {onReset && (
            <button type="button" onClick={onReset}
              title={t('workspace.crop.resetTitle')}
              className="mr-auto px-4 py-2 rounded-lg bg-surface text-content-muted text-sm">
              ↺ {t('workspace.crop.reset')}
            </button>
          )}
          <button type="button" ref={cancelRef} onClick={onCancel}
            className="px-4 py-2 rounded-lg bg-surface text-content text-sm">{t('common.close')}</button>
          <button type="button" disabled={!box}
            onClick={() => onConfirm({ x: Math.round(box.x), y: Math.round(box.y),
                                       w: Math.round(box.w), h: Math.round(box.h) })}
            className="px-4 py-2 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
            {t('workspace.crop.confirm')}
          </button>
        </div>
      </div>
    </div>
  );
}
