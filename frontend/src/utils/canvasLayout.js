/* Pure geometry for the ◉ LoRA Canvas — the board that puts EVERY dataset's
   lineage on one surface.

   Framework-free (no JSX, no DOM) so `node --test` exercises the real arithmetic
   the screen uses: stacking the datasets into lanes, fitting the board into its
   frame, and the zoom/pan bookkeeping. The renderer lives in
   components/canvas/LineageCanvas.jsx and imports exactly these functions — it
   owns no second copy of the maths.

   Two coordinate systems, and mixing them is the classic canvas bug:
     • WORLD  — the board's own units. A dataset lane is laid out here by
                utils/lineageGraph.js, untouched and unaware of zoom.
     • SCREEN — pixels inside the frame. screen = world * scale + translate.
   Every function below states which one it takes.

   Lanes stack in the order given. A lane with no PLACEMENT of its own sits at
   x = 0, under the one above it, and reserves exactly what its content needs —
   which is every lane on a board nobody has arranged. One that carries a
   placement (utils/canvasLanePlacement: an optional position, an optional
   reserved height) is drawn where it says and keeps the room it says, and the
   lanes below move with that room. This module consumes finished per-lane
   sizes and a finished placement; deciding either is somebody else's job. */

import { clampLanePlacement } from './canvasLanePlacement.js';

// A lane's title strip (dataset name + run count), above its graph.
export const LANE_HEADER_H = 34;
// Air between one dataset's board and the next. Generous on purpose: the whole
// point of the canvas is telling two datasets' genealogies apart at a glance.
export const LANE_GAP = 56;

// Zoom bounds. The floor is low enough that a library of a dozen trained
// datasets fits in one frame; the ceiling stops a stray pinch from blowing a
// 264-px card up to a wall.
export const MIN_SCALE = 0.1;
// 500% lets a generated image be inspected directly on the board. The former
// 250% ceiling was still too small after Fit had opened a large canvas at 10–50%.
export const MAX_SCALE = 5;

/** Clamp a scale into the usable range. NaN / nonsense degrades to 1 rather
 *  than blanking the board. */
export function clampScale(s) {
  const n = Number(s);
  if (!Number.isFinite(n)) return 1;
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, n));
}

/**
 * Stack per-dataset graphs into vertical lanes.
 *
 * `entries` — [{ datasetId, name, width, height, minX, minY, ... }] in the order
 * they should appear (the caller decides; the canvas keeps the filter's order).
 * A lane with no measurable graph still gets its header, so a dataset that is
 * still loading — or whose runs all vanished — keeps its place on the board
 * instead of making the lanes below it jump when it arrives.
 *
 * ⚠️ TWO EXTENTS, and telling them apart is the whole point of this function:
 *   • `width`/`height` are the lane's STACKING size — its tree, and nothing
 *     else. They alone decide where the next lane starts.
 *   • `minX`/`minY`/`maxX`/`maxY` are its content's REACH, lane-local, from
 *     utils/canvasImageNodes.imageNodeExtent: how far the pinned pictures get
 *     above/left (never positive) and right/below (defaulting to the lane's own
 *     size when absent). They grow only the BOX the board is framed, exported
 *     and clamped by — never the stack.
 *
 * That split is the fix for an asymmetry users hit immediately: a picture
 * dragged ABOVE its lane floated free, while the same picture dragged BELOW it
 * shoved the next dataset down the board, because the downward reach used to be
 * folded into `height` and `height` is what the stack advances by. Pinned
 * pictures are a free LAYER over the board, in both directions — at worst one
 * overlaps the lane below, which is what the person dragging it asked for.
 *
 * Returns { lanes, x, y, width, height } in WORLD units — the board's BOX, whose
 * top-left `x`/`y` may be negative. Each lane carries `y` (its header's top) and
 * `graphY` (its graph's top).
 *
 * ⚠️ Reach never MOVES a lane either — not down, not sideways. That was tried on
 * paper and it is the same bug from the other end: a picture dragged one pixel
 * past its lane's corner would move every lane on the board, so the whole thing
 * would slide under the hand still dragging it. Lanes keep the positions they
 * always had; only the box grows. With no reach fields anywhere, this returns
 * exactly the numbers it always returned, with x = y = 0.
 */
export function stackLanes(entries) {
  const list = Array.isArray(entries) ? entries : [];
  const lanes = [];
  let y = 0;
  let left = 0;
  let top = 0;
  let right = 0;
  let bottom = 0;
  for (const e of list) {
    const w = Math.max(0, Number(e?.width) || 0);
    const h = Math.max(0, Number(e?.height) || 0);
    const overX = Math.min(0, Number(e?.minX) || 0);
    const overY = Math.min(0, Number(e?.minY) || 0);
    const reachX = Math.max(w, Number(e?.maxX) || 0);
    /* What the lane actually DRAWS below its own origin, and what it KEEPS.
       They were the same number until lanes could be arranged, and conflating
       them is the collision this split exists to end: a 📌 Pin all band reaches
       far below the tree, the stack only ever advanced by the tree, so the band
       landed on the next dataset (measured: 894 units of the lane below).
       `reserved` is now what the user said, when they said anything. */
    const contentH = Math.max(h, Number(e?.maxY) || 0);
    const placement = clampLanePlacement(e?.placement);
    const reserved = placement?.h != null ? placement.h : h;
    // ✦ Fit and 📷 Export frame the REACH, never the reservation: a lane whose
    // content overflows the room it keeps is still framed and still exported —
    // it just overlaps, which is what the person who set that height chose.
    const reachY = Math.max(reserved, contentH);
    const laneX = placement?.x != null ? placement.x : 0;
    const laneY = placement?.y != null ? placement.y : y;
    const lane = { ...e, x: laneX, y: laneY, graphY: laneY + LANE_HEADER_H,
      width: w, height: h, reserved, contentH, placement: placement || null };
    lanes.push(lane);
    left = Math.min(left, lane.x + overX);
    top = Math.min(top, lane.graphY + overY);
    right = Math.max(right, lane.x + reachX);
    // The lane's bottom, header included: `graphY` already carries the header,
    // so a lane with no graph still contributes its title strip. This is also
    // why there is no trailing gap — the board ends where the content ends,
    // instead of carrying one lane-gap of dead space "fit" would waste.
    bottom = Math.max(bottom, lane.graphY + reachY);
    /* …and the STACK advances by the room the lane KEEPS. `contentH` is
       deliberately absent here: that is what lets a picture hang below its lane
       without pushing the next dataset down the board — the rule that stops the
       whole board sliding under the hand still dragging the picture.

       ⚠️ A MOVED lane still advances the cursor. Taking it out of the flow
       instead would make every other lane jump the moment one is dragged, which
       is the same bug from the other end. Moving one lane moves exactly one
       lane; where it lands is then its owner's business, like a dragged card. */
    y += LANE_HEADER_H + reserved + LANE_GAP;
  }
  if (!lanes.length) return { lanes, x: 0, y: 0, width: 0, height: 0 };
  return { lanes, x: left, y: top, width: right - left, height: bottom - top };
}

/** The board's top-left corner in WORLD units. Zero for a board with nothing
 *  hanging out of its lanes — which is every board that predates free placement
 *  — so a caller that only passes {width,height} gets the maths it always got. */
const worldOrigin = (world) => ({
  x: Number(world?.x) || 0,
  y: Number(world?.y) || 0,
});

/**
 * The view that shows the whole board inside `viewport`, centred.
 * `world` {x,y,width,height} and `viewport` {width,height} are both in their own
 * units; the result is { scale, tx, ty } with screen = world * scale + t.
 *
 * `world.x`/`world.y` may be NEGATIVE — a picture parked above or left of its
 * lane puts the board's corner there — and the translate carries them, so what
 * gets centred is the board's real box rather than the quadrant below its
 * origin. Without that, the one gesture free placement exists for (drag a render
 * up, above its lane) would produce a picture ✦ Fit could not bring back.
 *
 * Never magnifies past 1 — a two-run lineage blown up to fill a 27" screen
 * looks broken, not helpful. An empty board or an unmeasured frame answers the
 * identity view instead of dividing by zero.
 */
export function fitView(world, viewport, { padding = 16 } = {}) {
  const ww = Math.max(0, Number(world?.width) || 0);
  const wh = Math.max(0, Number(world?.height) || 0);
  const vw = Math.max(0, Number(viewport?.width) || 0);
  const vh = Math.max(0, Number(viewport?.height) || 0);
  if (!ww || !wh || !vw || !vh) return { scale: 1, tx: padding, ty: padding };
  const availW = Math.max(1, vw - padding * 2);
  const availH = Math.max(1, vh - padding * 2);
  const scale = clampScale(Math.min(1, Math.min(availW / ww, availH / wh)));
  const origin = worldOrigin(world);
  return {
    scale,
    tx: (vw - ww * scale) / 2 - origin.x * scale,
    ty: (vh - wh * scale) / 2 - origin.y * scale,
  };
}

// The smallest scale the board is allowed to OPEN at. A true fit is the right
// answer for the Fit button, but not for a first paint on a phone: a board of
// three datasets fits a 400-px frame at ~35 %, where a run card is four pixels
// tall and nothing is readable. Opening a little closer and letting the user
// scroll down beats opening on an unreadable overview.
export const INITIAL_MIN_SCALE = 0.45;

/**
 * The view the board OPENS at: the fit, but never below INITIAL_MIN_SCALE, and
 * ALWAYS top-aligned.
 *
 * Top-aligned even when the board is shorter than its frame. Vertically
 * centring a short board looked tidy on a desktop and was expensive on a phone:
 * the frame is sized from the viewport, not from the content, so a two-lane
 * board opened with a third of the frame as empty sky above the first lane —
 * a third of a 400-px screen spent on nothing before anything is legible.
 * Horizontal centring stays: the lanes are narrower than they are tall and
 * there is no equivalent waste.
 */
export function initialView(world, viewport, { padding = 16 } = {}) {
  const fit = fitView(world, viewport, { padding });
  const ww = Math.max(0, Number(world?.width) || 0);
  const wh = Math.max(0, Number(world?.height) || 0);
  const vw = Math.max(0, Number(viewport?.width) || 0);
  const vh = Math.max(0, Number(viewport?.height) || 0);
  if (!ww || !wh || !vw || !vh) return fit;
  const scale = clampScale(Math.min(1, Math.max(fit.scale, INITIAL_MIN_SCALE)));
  const w = ww * scale;
  // Same origin correction as fitView: "top-aligned" means the top of the
  // board's BOX, which is above the first lane when something hangs over it.
  const origin = worldOrigin(world);
  return {
    scale,
    tx: (w <= vw ? (vw - w) / 2 : padding) - origin.x * scale,
    ty: padding - origin.y * scale,
  };
}

/** WORLD point under a SCREEN point, for the current view. */
export function toWorld(view, sx, sy) {
  const s = clampScale(view?.scale);
  return { x: (sx - (view?.tx || 0)) / s, y: (sy - (view?.ty || 0)) / s };
}

/**
 * Zoom by `factor` around a SCREEN anchor — the point the user is pointing at
 * stays exactly where it is, which is what makes wheel-zoom and pinch feel like
 * they grab the board rather than shove it. Clamping the scale must not move
 * the anchor either, so the translate is recomputed from the CLAMPED scale.
 */
export function zoomAt(view, factor, anchor) {
  const s0 = clampScale(view?.scale);
  const s1 = clampScale(s0 * (Number(factor) || 1));
  const ax = Number(anchor?.x) || 0;
  const ay = Number(anchor?.y) || 0;
  const w = toWorld({ ...view, scale: s0 }, ax, ay);
  return { scale: s1, tx: ax - w.x * s1, ty: ay - w.y * s1 };
}

/** Pan by a SCREEN delta. */
export function panBy(view, dx, dy) {
  return {
    scale: clampScale(view?.scale),
    tx: (Number(view?.tx) || 0) + (Number(dx) || 0),
    ty: (Number(view?.ty) || 0) + (Number(dy) || 0),
  };
}

/**
 * Keep the board reachable: after any pan or zoom, at least `keep` screen pixels
 * of content must remain inside the frame on each axis. Without this the board
 * can be flung off-screen by one careless swipe and the only way back is a
 * reload — the failure mode every hand-written canvas ships with first.
 */
export function clampView(view, world, viewport, { keep = 80 } = {}) {
  const scale = clampScale(view?.scale);
  const ww = (Math.max(0, Number(world?.width) || 0)) * scale;
  const wh = (Math.max(0, Number(world?.height) || 0)) * scale;
  const vw = Math.max(0, Number(viewport?.width) || 0);
  const vh = Math.max(0, Number(viewport?.height) || 0);
  if (!ww || !wh || !vw || !vh) return { scale, tx: Number(view?.tx) || 0, ty: Number(view?.ty) || 0 };
  const clamp1 = (t, content, frame) => {
    const margin = Math.min(keep, content);
    return Math.min(frame - margin, Math.max(margin - content, t));
  };
  // The clamp is about where the CONTENT sits on screen, so it is applied to
  // the board box's own left/top edge — which is `tx` shifted by the world
  // origin — and undone afterwards. A board whose corner is at (0,0), which is
  // every board with nothing hanging out of a lane, is clamped exactly as
  // before.
  const origin = worldOrigin(world);
  const ox = origin.x * scale;
  const oy = origin.y * scale;
  return {
    scale,
    tx: clamp1((Number(view?.tx) || 0) + ox, ww, vw) - ox,
    ty: clamp1((Number(view?.ty) || 0) + oy, wh, vh) - oy,
  };
}

/** The CSS transform for a view. `transform-origin: 0 0` is assumed — the order
 *  (translate then scale) is the same one toWorld inverts. */
export function viewTransform(view) {
  const s = clampScale(view?.scale);
  return `translate(${Number(view?.tx) || 0}px, ${Number(view?.ty) || 0}px) scale(${s})`;
}

/** Distance between two pointers, for pinch zoom. */
export function pinchDistance(a, b) {
  const dx = (Number(a?.x) || 0) - (Number(b?.x) || 0);
  const dy = (Number(a?.y) || 0) - (Number(b?.y) || 0);
  return Math.hypot(dx, dy);
}

/** Midpoint of two pointers — the anchor a pinch zooms around. */
export function pinchCenter(a, b) {
  return {
    x: ((Number(a?.x) || 0) + (Number(b?.x) || 0)) / 2,
    y: ((Number(a?.y) || 0) + (Number(b?.y) || 0)) / 2,
  };
}
