/* Where a LANE sits on the ◉ LoRA Canvas, and how much room it keeps.

   Everything else on this board can be arranged. A run card remembers where it
   was dragged (utils/canvasPlacement), a pinned picture remembers its box
   (utils/canvasImageNodes), a strip of pictures is moved by its bar and resized
   by its corner. The lane — the dataset's own block, its title strip and
   everything under it — was the one object with neither gesture: `stackLanes`
   pinned it to x = 0 and stacked it by its TREE's height alone.

   That was not only a missing affordance, it was a measured collision. A lane
   advances the stack by `LANE_HEADER_H + tree height + LANE_GAP`, while
   📌 Pin all lays its contact sheet BELOW the tree — so the sheet lands on the
   next dataset's header, cards and pictures. Measured on a two-lane board with
   a four-row band: 894 world units of the lane below covered.

   The deliberate rule it collides with is right and stays: a picture dragged
   under its lane must NOT push the next dataset down the board, or the board
   would slide under the hand still dragging. What was missing is the other half
   — a way to SAY how much room a lane keeps. That is what a placement is.

   ── What a placement is ─────────────────────────────────────────────────────
   Three independent, all optional:

     • `h` — the room the lane RESERVES, header excluded. Drives the stack, so
       the lanes below move when it changes. Absent means "whatever the content
       needs", exactly as before.
     • `x`/`y` — where the lane is DRAWN, in board units. Absent means "where the
       stack puts it".

   `x`/`y` do not remove the lane from the stack: the cursor still advances by
   its reserved height, so moving one lane never reshuffles the others. A lane
   parked on top of another is then the user's own doing — the same bargain a
   dragged card and a dragged picture already make, and ✦ Tidy up is the way
   back from all three.

   Pure functions, no JSX and no DOM: `node --test` exercises the arithmetic the
   screen uses, which is the only part of this that can silently rot. */

// The floor. A lane shorter than its own title strip could not be grabbed to be
// made bigger again — the trap a resize handle must never be able to build.
export const LANE_MIN_H = 96;
// The ceiling, and the reason for it is ✦ Fit: the board is framed by its total
// box, so one lane reserving a hundred thousand units would collapse every
// other lane to a scale where nothing is readable. Same job as IMG_MAX.
export const LANE_MAX_H = 40000;
/* How far from the board's origin a lane may be parked, on either axis and in
   either direction. A SAFETY RAIL, not a design limit — same reasoning and the
   same number as IMG_REACH in utils/canvasImageNodes: one corrupt row (1e9, a
   hand-edited database) must not be able to make ✦ Fit collapse the board, and
   there is no UI to fix a lane parked at NaN. */
export const LANE_REACH = 100000;

const reach = (v) => Math.min(LANE_REACH, Math.max(-LANE_REACH, v));

/** A finite number, or null. Anything else — NaN, Infinity, '', undefined — is
 *  ABSENT rather than zero: a lane silently teleported to the board's corner is
 *  worse than a lane the stack places itself. */
const finiteOrNull = (v) => {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

/**
 * Normalise one placement into what the layout may consume, or null.
 *
 * Returns null when NOTHING usable is left — which is what "this lane has no
 * placement" means everywhere else, so a row of three nulls read back from a
 * database behaves exactly like no row at all.
 */
export function clampLanePlacement(placement) {
  if (!placement || typeof placement !== 'object') return null;
  const x = finiteOrNull(placement.x);
  const y = finiteOrNull(placement.y);
  const h = finiteOrNull(placement.h);
  const out = {};
  if (x != null) out.x = reach(x);
  if (y != null) out.y = reach(y);
  if (h != null) out.h = Math.min(LANE_MAX_H, Math.max(LANE_MIN_H, h));
  // x and y travel TOGETHER: half a position is not a position, and a lane with
  // a y but no x would sit at the board's left edge for reasons no one could
  // read off the row.
  if ((out.x == null) !== (out.y == null)) {
    delete out.x;
    delete out.y;
  }
  return Object.keys(out).length ? out : null;
}

/**
 * The API's rows into {datasetId: placement}. Rows that normalise to nothing are
 * DROPPED, so a lane whose row says "auto, auto, auto" is indistinguishable
 * from a lane with no row — one state, not two.
 */
export function toLanePlacementMap(rows) {
  const out = {};
  for (const r of (Array.isArray(rows) ? rows : [])) {
    const id = r?.dataset_id ?? r?.datasetId;
    if (id == null) continue;
    const placement = clampLanePlacement(r);
    if (placement) out[String(id)] = placement;
  }
  return out;
}

/**
 * What the lane's bottom edge is being dragged TO, as a reserved height.
 *
 * `startH` is the height at the moment the drag began — the reserved one if the
 * lane already had one, the automatic one otherwise, so grabbing the edge of a
 * never-touched lane does not make it jump to a floor first.
 */
export function resizeLaneHeight(startH, dy) {
  const base = Number(startH);
  const delta = Number(dy);
  const next = (Number.isFinite(base) ? base : LANE_MIN_H)
    + (Number.isFinite(delta) ? delta : 0);
  return Math.min(LANE_MAX_H, Math.max(LANE_MIN_H, next));
}

/**
 * Move a lane by (dx, dy) from where it currently sits.
 *
 * The lane's CURRENT box is passed in rather than its placement, because a lane
 * that has never been moved has no placement to add a delta to — its position
 * is wherever the stack put it, and that is exactly the position the drag must
 * start from so the block does not jump under the pointer on the first pixel.
 */
export function moveLaneTo(box, dx, dy) {
  const x = Number(box?.x);
  const y = Number(box?.y);
  return {
    x: reach((Number.isFinite(x) ? x : 0) + (Number(dx) || 0)),
    y: reach((Number.isFinite(y) ? y : 0) + (Number(dy) || 0)),
  };
}

/**
 * The placement a gesture leaves behind: what the lane had, plus what the
 * gesture changed. A move must not forget a reserved height, and a resize must
 * not forget a position — they are three independent facts about one lane and
 * each gesture only ever speaks for its own.
 */
export function mergeLanePlacement(current, change) {
  return clampLanePlacement({ ...(current || {}), ...(change || {}) });
}

/**
 * Is this lane's reserved room SHORTER than what it actually draws?
 *
 * The whole reason the feature exists, so it is a function rather than a
 * comparison written at the call site: it is what the resize handle uses to
 * offer "fit to content", and what a test asserts against the placement the
 * band produces. `content` is the lane's real reach below its own origin
 * (tree, strips and contact-sheet band included).
 */
export function laneOverflows(lane) {
  const reserved = Number(lane?.reserved);
  const content = Number(lane?.contentH);
  if (!Number.isFinite(reserved) || !Number.isFinite(content)) return false;
  return content > reserved;
}
