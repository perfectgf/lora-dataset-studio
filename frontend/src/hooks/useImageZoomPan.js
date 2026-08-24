import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  FIT_VIEW, clampPan, doubleTapView, fitSize, isZoomed, maxZoomFor, panByDelta,
  tapOutcome, zoomAtPoint,
} from '../utils/imageZoomPan';

/* 🔍 WHICH FINGER DID WHAT — the gestures, over utils/imageZoomPan's geometry.
 *
 * Pointer Events and not touch handlers: one code path for a finger, a mouse
 * and a pen, and `setPointerCapture` means a drag that leaves the picture keeps
 * being a drag instead of ending wherever the browser lost interest.
 *
 * ⚠️ THE ONE HARD PART IS THE TAP, because this viewer already had one. A
 * single tap folds the details away; a double tap now zooms. They start
 * identically, so the single tap has to WAIT to find out which it was —
 * DOUBLE_TAP_MS of it. That delay is real and it is the price: firing the fold
 * immediately would make every double tap fold the panel first and zoom second,
 * which reads as the viewer flinching. 260 ms is short enough to feel like a
 * response and long enough for a second tap that is actually coming.
 *
 * The tap only counts on the PICTURE. The frame around it belongs to the
 * viewer's backdrop, where a click has always meant close, and taking that over
 * would remove the way out that people reach for first — so the hook reports
 * where the press started and the component decides.
 */

/** How long a first tap waits to find out whether it is half of a double one. */
export const DOUBLE_TAP_MS = 260;

/** One wheel notch. Small enough to feel continuous on a trackpad, big enough
 *  that a mouse wheel gets somewhere in three clicks. */
const WHEEL_STEP = 1.18;

/**
 * @param {object} opts
 *  - imgRef, frameRef: the picture and the box it is drawn in.
 *  - active: false while there is nothing on screen — every listener and all
 *    the state stand down rather than being conditionally attached.
 *  - resetKey: change it (the image URL) and the view goes back to fit. A new
 *    picture inherited at 3x, panned to a corner of the previous one, is the
 *    bug this exists to prevent.
 *  - onTap: what a confirmed single tap on the picture means to the host.
 */
export function useImageZoomPan({ imgRef, frameRef, active = true, resetKey = null, onTap = null }) {
  const [view, setView] = useState(FIT_VIEW);
  // Live pointers, by id. A Map rather than an array so a pointer that ends out
  // of order (a finger lifted mid-pinch) removes exactly itself.
  const pointers = useRef(new Map());
  const gesture = useRef(null);
  const tapTimer = useRef(null);
  const viewRef = useRef(view);
  viewRef.current = view;

  useEffect(() => () => clearTimeout(tapTimer.current), []);
  // Back to fit whenever the picture changes or the viewer closes.
  useEffect(() => {
    setView(FIT_VIEW);
    pointers.current.clear();
    gesture.current = null;
    clearTimeout(tapTimer.current);
  }, [resetKey, active]);

  /* The measurements every rule needs, taken at gesture time rather than kept
     in state: they change with the window, with the facts panel folding, and
     with the picture finishing loading, and a stale box is a picture that
     cannot reach its own corner. */
  const readBox = useCallback(() => {
    const frame = frameRef.current?.getBoundingClientRect();
    const img = imgRef.current;
    if (!frame || !img) return null;
    const fit = fitSize(img.naturalWidth, img.naturalHeight, frame.width, frame.height);
    if (!fit.width) return null;
    return {
      fitW: fit.width, fitH: fit.height, frameW: frame.width, frameH: frame.height,
      cx: frame.left + frame.width / 2, cy: frame.top + frame.height / 2,
      natural: img.naturalWidth,
    };
  }, [frameRef, imgRef]);

  /** A client point in the frame-centred coordinates the geometry speaks. */
  const toLocal = (box, clientX, clientY) => ({ x: clientX - box.cx, y: clientY - box.cy });

  const maxZoom = useCallback((box) => maxZoomFor(box.natural, box.fitW), []);

  const zoomBy = useCallback((factor, clientX, clientY) => {
    const box = readBox();
    if (!box) return;
    setView((v) => zoomAtPoint(v, factor, toLocal(box, clientX, clientY), box, maxZoom(box)));
  }, [readBox, maxZoom]);

  const reset = useCallback(() => setView(FIT_VIEW), []);

  /* Re-apply rule 1 without touching the zoom. The frame changes size under a
     held view every time the details fold, the window resizes or the picture
     finishes loading — and a view that was legally at its edge before is a gap
     of background afterwards. Nothing here moves a picture the user placed;
     it only pulls back one that stopped being reachable. */
  const settle = useCallback(() => {
    const box = readBox();
    if (box) setView((v) => clampPan(v, box));
  }, [readBox]);

  const onPointerDown = useCallback((e) => {
    if (!active) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    // Spec: capturing a pointerId that is not active throws NotFoundError. That
    // happens with synthetic events and with a pointer the browser has already
    // released, and an exception here would take the whole gesture down.
    try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { /* not capturable */ }
    const box = readBox();
    const live = [...pointers.current.values()];
    if (live.length === 2 && box) {
      // A pinch starts from a snapshot: every move is measured against the
      // distance and the view the fingers STARTED at, never against the last
      // frame. Accumulating per-frame ratios drifts, and drift in a pinch is a
      // picture that ends up somewhere you did not put it.
      clearTimeout(tapTimer.current);
      const [a, b] = live;
      gesture.current = {
        kind: 'pinch',
        startDist: Math.hypot(a.x - b.x, a.y - b.y) || 1,
        startView: viewRef.current,
      };
      return;
    }
    gesture.current = {
      kind: 'press',
      id: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      lastX: e.clientX,
      lastY: e.clientY,
      at: e.timeStamp,
      moved: 0,
      // Where the press LANDED decides whether its tap belongs to the viewer or
      // to the backdrop behind it.
      onImage: e.target === imgRef.current,
    };
  }, [active, readBox, imgRef]);

  const onPointerMove = useCallback((e) => {
    if (!active || !pointers.current.has(e.pointerId)) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const g = gesture.current;
    const box = readBox();
    if (!g || !box) return;
    if (g.kind === 'pinch') {
      const live = [...pointers.current.values()];
      if (live.length < 2) return;
      const [a, b] = live;
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const mid = toLocal(box, (a.x + b.x) / 2, (a.y + b.y) / 2);
      setView(zoomAtPoint(g.startView, dist / g.startDist, mid, box, maxZoom(box)));
      return;
    }
    if (g.kind !== 'press' || g.id !== e.pointerId) return;
    const dx = e.clientX - g.lastX;
    const dy = e.clientY - g.lastY;
    g.moved += Math.hypot(dx, dy);
    g.lastX = e.clientX;
    g.lastY = e.clientY;
    // A drag only means something when there is somewhere to go. At fit the
    // press stays a candidate tap however far it wanders, which is what lets a
    // swipe across an unzoomed picture do nothing instead of nudging it.
    if (isZoomed(viewRef.current)) setView((v) => panByDelta(v, dx, dy, box));
  }, [active, readBox, maxZoom]);

  const endPointer = useCallback((e, cancelled) => {
    pointers.current.delete(e.pointerId);
    const g = gesture.current;
    if (!g) return;
    if (g.kind === 'pinch') {
      // The second finger going up hands the gesture back to the first as a
      // fresh press, so a pinch that relaxes into a drag keeps working.
      const live = [...pointers.current.entries()];
      /* ⚠️ …as a press that can never become a TAP. Measured in a browser: with
         the finger handed over plain, letting go of the second one ended a
         gesture that had moved 0 px in 0 ms, which is the exact shape of a tap —
         so every pinch finished by folding the details away. A finger that was
         half of a pinch is finishing a pinch, whatever it does next. */
      gesture.current = live.length === 1
        ? { kind: 'press', id: live[0][0], startX: live[0][1].x, startY: live[0][1].y,
            lastX: live[0][1].x, lastY: live[0][1].y, at: e.timeStamp, moved: 0,
            onImage: true, fromPinch: true }
        : null;
      // A pinch can leave the picture slightly outside its travel if the box
      // changed under it; settle it rather than leaving a gap on screen.
      const box = readBox();
      if (box && live.length === 0) setView((v) => clampPan(v, box));
      return;
    }
    if (g.kind !== 'press' || g.id !== e.pointerId) return;
    gesture.current = null;
    if (cancelled) return;
    // What that press meant is a decision, not a gesture — utils/imageZoomPan
    // owns it, and `node --test` holds it to the browser's own verdict.
    const outcome = tapOutcome(
      { moved: g.moved, held: e.timeStamp - g.at, onImage: g.onImage, fromPinch: g.fromPinch },
      { pendingTap: !!tapTimer.current },
    );
    if (outcome === 'ignore') return;

    if (outcome === 'double') {
      // The second tap of a pair: the first one's fold never happens.
      clearTimeout(tapTimer.current);
      tapTimer.current = null;
      const box = readBox();
      if (box) setView((v) => doubleTapView(v, toLocal(box, e.clientX, e.clientY), box, maxZoom(box)));
      return;
    }
    tapTimer.current = setTimeout(() => {
      tapTimer.current = null;
      onTap?.();
    }, DOUBLE_TAP_MS);
  }, [readBox, maxZoom, onTap]);

  const onPointerUp = useCallback((e) => endPointer(e, false), [endPointer]);
  const onPointerCancel = useCallback((e) => endPointer(e, true), [endPointer]);

  /* The wheel is attached by hand, `{ passive: false }`: React's onWheel is
     passive, so preventDefault() inside it does nothing and the browser zooms
     the whole PAGE over a viewer that was already zooming the picture. */
  useEffect(() => {
    const el = frameRef.current;
    if (!el || !active) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      zoomBy(e.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP, e.clientX, e.clientY);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [frameRef, active, zoomBy]);

  const zoomed = isZoomed(view);

  return useMemo(() => ({
    view,
    zoomed,
    reset,
    settle,
    zoomBy,
    style: {
      transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`,
      transformOrigin: 'center',
      // No transition: a transform that eases lags a finger by its own duration,
      // and a magnifier that lags is a magnifier you cannot aim.
      cursor: zoomed ? 'grab' : 'auto',
      willChange: zoomed ? 'transform' : 'auto',
    },
    handlers: { onPointerDown, onPointerMove, onPointerUp, onPointerCancel },
  }), [view, zoomed, reset, settle, zoomBy,
    onPointerDown, onPointerMove, onPointerUp, onPointerCancel]);
}

export default useImageZoomPan;
