/* 🔍 ZOOMING A PICTURE INSIDE A VIEWER — the geometry, and nothing else.
 *
 * WHY THIS EXISTS, measured rather than argued. The generated-image viewer can
 * now fold its details away and give the picture the whole window, and on every
 * shape but one that was the answer: a tablet held sideways went from 35 % of
 * the screen to 90 %, a desktop window from 56 % to 84 %. A phone held UPRIGHT
 * went from 35 % to 39 % and stopped there — and the reason is not the panel,
 * it is the aspect ratio. A 4:3 render at 412 CSS px wide is drawn 412x309
 * whatever else is on screen: it already has every pixel of the scarce axis.
 * Folding cannot give it more. Only magnifying can.
 *
 * So this is the other half of "let me see the render properly", and it is the
 * half that a phone actually needs.
 *
 * ── THE MODEL ───────────────────────────────────────────────────────────────
 * `scale` is a multiple of the FIT size, not of the file's pixels. 1 means "the
 * picture as `object-contain` draws it", which is where the viewer opens and
 * what every reset returns to. That choice is what keeps this module ignorant
 * of the image's own resolution everywhere except `maxZoomFor`.
 *
 * `tx`/`ty` are CSS pixels of translation applied around the picture's CENTRE
 * (`transform-origin: center`), because that is where a centred `object-contain`
 * image already sits — measuring from a corner would mean re-deriving the
 * centring on every call.
 *
 * ── THE TWO RULES A PHOTO VIEWER OWES YOU ───────────────────────────────────
 *  1. THE PICTURE NEVER RUNS AWAY. Zoomed in, no pan may show background where
 *     the picture could be: the travel on each axis is exactly the overflow,
 *     `(drawn - frame) / 2`, and zero when the picture is smaller than the
 *     frame. This is the failure every hand-written pan ships with first — one
 *     careless swipe and the picture is off-screen with no way back but a
 *     reload.
 *  2. ZOOM STOPS WHERE THE DETAIL DOES. Past the file's own resolution you are
 *     magnifying interpolation, and a viewer that lets you do that is lying
 *     about what it can show — the whole reason to zoom a render is to judge
 *     whether the model got an eye or a hand right. `maxZoomFor` caps at native
 *     pixels, with a floor so that a small file is still inspectable and a
 *     ceiling so that a huge one does not offer a 40x nobody can aim.
 *
 * Pure and DOM-free, so `node --test` can hold it to those two rules directly.
 * The gesture wiring — which finger did what — is hooks/useImageZoomPan.js.
 */

/** Fit. The view a freshly opened picture is at, and what every reset returns. */
export const FIT_VIEW = Object.freeze({ scale: 1, tx: 0, ty: 0 });

/** Never below fit: a viewer that can shrink the picture inside its own window
 *  offers a state with no use and a lot of empty black. */
export const MIN_ZOOM = 1;

/** A small file must still be inspectable, so the cap never falls under this
 *  even when the picture is already drawn at its native size. */
export const MIN_MAX_ZOOM = 2;

/** …and a 6000-px render must not offer a 15x nobody can aim on a phone. */
export const MAX_MAX_ZOOM = 8;

/** What one double-tap jumps to, bounded by the cap. Not the maximum: a double
 *  tap is "closer", not "as close as possible", and landing at the far end
 *  leaves the gesture nowhere to go. */
export const DOUBLE_TAP_ZOOM = 3;

const num = (v, fallback = 0) => (Number.isFinite(Number(v)) ? Number(v) : fallback);
/* `Math.max(-0, -900)` is -0, and -0 travels: it reaches the transform string,
   any snapshot and any equality check as a value that is not 0 while being
   equal to it. A centred picture must report a centred picture. */
const noNegZero = (n) => (n === 0 ? 0 : n);

/**
 * How far in this picture is worth going: the point at which one drawn pixel is
 * one file pixel.
 *
 * `naturalPx` is the file's own width, `fitPx` the width it is drawn at. A
 * picture already drawn larger than its file (a small render on a big screen)
 * returns the floor rather than something under 1 — "you may not zoom at all"
 * is never the right answer from a magnifier.
 */
export function maxZoomFor(naturalPx, fitPx) {
  const natural = num(naturalPx, 0);
  const fit = num(fitPx, 0);
  if (natural <= 0 || fit <= 0) return MIN_MAX_ZOOM;
  return Math.min(MAX_MAX_ZOOM, Math.max(MIN_MAX_ZOOM, natural / fit));
}

/** Hold a scale inside [MIN_ZOOM, max]. */
export function clampZoom(scale, max = MAX_MAX_ZOOM) {
  const top = Math.max(MIN_ZOOM, num(max, MAX_MAX_ZOOM));
  return Math.min(top, Math.max(MIN_ZOOM, num(scale, 1)));
}

/**
 * Rule 1, alone and testable: pull a view back until the picture covers the
 * frame on both axes.
 *
 * `box` — { fitW, fitH, frameW, frameH } in CSS px: the size `object-contain`
 * draws the picture at, and the box it is drawn in.
 */
export function clampPan(view, box) {
  const scale = clampZoom(view?.scale, MAX_MAX_ZOOM);
  const drawnW = Math.max(0, num(box?.fitW)) * scale;
  const drawnH = Math.max(0, num(box?.fitH)) * scale;
  // The travel IS the overflow, halved — the picture is centred, so it can move
  // by as much as sticks out on either side and not one pixel more. Nothing
  // sticking out means no travel at all, which is what re-centres a picture the
  // moment it is zoomed back to fit.
  const maxX = Math.max(0, (drawnW - Math.max(0, num(box?.frameW))) / 2);
  const maxY = Math.max(0, (drawnH - Math.max(0, num(box?.frameH))) / 2);
  return {
    scale,
    tx: noNegZero(Math.min(maxX, Math.max(-maxX, num(view?.tx)))),
    ty: noNegZero(Math.min(maxY, Math.max(-maxY, num(view?.ty)))),
  };
}

/**
 * Zoom by `factor` while keeping whatever is under `point` under `point`.
 *
 * `point` is in FRAME coordinates with the origin at the frame's centre — the
 * same origin the transform uses, so no centring arithmetic leaks into the
 * caller. Anchoring is the whole difference between a magnifier and a control
 * that throws away what you were looking at every time you use it: pinch on a
 * face, and the face is what gets bigger.
 */
export function zoomAtPoint(view, factor, point, box, max = MAX_MAX_ZOOM) {
  const s0 = clampZoom(view?.scale, max);
  const s1 = clampZoom(s0 * num(factor, 1), max);
  const px = num(point?.x);
  const py = num(point?.y);
  const tx0 = num(view?.tx);
  const ty0 = num(view?.ty);
  // The point's offset within the picture is (p - t) / s; putting it back under
  // the same p at the new scale gives p - s1/s0 * (p - t).
  const k = s1 / (s0 || 1);
  return clampPan({ scale: s1, tx: px - k * (px - tx0), ty: py - k * (py - ty0) }, box);
}

/** Pan by a screen delta, then rule 1. */
export function panByDelta(view, dx, dy, box) {
  return clampPan({
    scale: view?.scale,
    tx: num(view?.tx) + num(dx),
    ty: num(view?.ty) + num(dy),
  }, box);
}

/**
 * What a double tap does: in if you are at fit, all the way back out if you are
 * not.
 *
 * Out is unconditional and out is to FIT, never one step — a gesture whose job
 * is "put it back" must put it back in one go, from any zoom, or the way home
 * is a number of taps you have to count.
 */
export function doubleTapView(view, point, box, max = MAX_MAX_ZOOM) {
  const scale = clampZoom(view?.scale, max);
  if (scale > MIN_ZOOM + 1e-6) return { ...FIT_VIEW };
  const target = clampZoom(DOUBLE_TAP_ZOOM, max);
  return zoomAtPoint({ ...FIT_VIEW }, target, point, box, max);
}

/**
 * What a finished press meant. The one decision in this file that is not
 * geometry, and it is here rather than in the hook because it is the one that
 * was got WRONG in a browser and has to stay got right.
 *
 * `press` — { moved, held, onImage, fromPinch }: screen px travelled, ms held,
 * whether it started on the picture, and whether it is the last finger of a
 * pinch. `pendingTap` — is a first tap already waiting to find out what it was.
 *
 * ⚠️ `fromPinch` is the one that was measured rather than reasoned. When a pinch
 * ends, the surviving finger is handed back as an ordinary press — and letting
 * go of it ends a press that has moved 0 px in 0 ms, which is byte-for-byte the
 * shape of a tap. Every pinch therefore finished by folding the details away.
 * A finger that was half of a pinch is finishing a pinch, whatever it does next.
 */
export function tapOutcome(press, { pendingTap = false } = {}) {
  if (!press || !press.onImage || press.fromPinch) return 'ignore';
  if (num(press.moved) > TAP_SLOP_PX || num(press.held) > TAP_MAX_MS) return 'ignore';
  return pendingTap ? 'double' : 'single';
}

/** Screen pixels of travel that turn a press into a drag rather than a tap. A
 *  thumb never lands perfectly still; 8 px is the wobble, not an intent. */
export const TAP_SLOP_PX = 8;

/** Longer than this and the press was a hold, not a tap. */
export const TAP_MAX_MS = 500;

/** Is this view magnified at all? Below this the pointer belongs to the tap
 *  gestures, not to a pan — there is nowhere to pan to. */
export function isZoomed(view) {
  return clampZoom(view?.scale, MAX_MAX_ZOOM) > MIN_ZOOM + 1e-6;
}

/**
 * The size `object-contain` draws a picture at inside a frame — the one piece
 * of geometry the browser knows and never tells you, and every rule above needs
 * it. Returns zeros for an unmeasurable picture so a caller can tell "not laid
 * out yet" from "one pixel wide".
 */
export function fitSize(naturalW, naturalH, frameW, frameH) {
  const nw = Math.max(0, num(naturalW));
  const nh = Math.max(0, num(naturalH));
  const fw = Math.max(0, num(frameW));
  const fh = Math.max(0, num(frameH));
  if (!nw || !nh || !fw || !fh) return { width: 0, height: 0 };
  const k = Math.min(fw / nw, fh / nh);
  return { width: nw * k, height: nh * k };
}
