/* The CHROME of a pinned image on the ◉ LoRA Canvas — its ✕, its 🔍 and its
   resize corner — and why those three need arithmetic of their own.

   Everything else on this board is measured in BOARD units, and that is right:
   a card, a pill and a picture all live in one zoomable coordinate system, so
   they keep their relationships at every scale. Controls are the exception, and
   the bug report is the proof. The ✕ of a pinned image "did not work" on a
   phone. The handler was fine — the board has refused to start a drag from a
   node's own button for as long as the feature has existed. What was not fine
   was the TARGET: the header button is ~16 board units, the board is routinely
   read at 24-65 %, and 16 × 0.65 ≈ 10 screen pixels — a quarter of the ~40 px a
   finger actually lands on. The button was not broken, it was unhittable, and
   the 🔍 sitting right beside it meant a near miss opened the full-screen view
   instead of closing the node. That reads exactly like "the cross does nothing".

   So a control is sized in SCREEN space: it is counter-scaled by the board's own
   zoom, which keeps it the same size under the finger whatever the board is
   doing. Two bounds keep that honest:

     • never SMALLER than its board size (zooming in must not shrink the chrome
       into a speck sitting on a huge picture);
     • never so large that it eats the picture it decorates — at 10 % zoom a
       constant-size button would be wider than the thumbnail it sits on, so the
       counter-scale is capped as a fraction of the node's own width.

   JSX-free, because `node --test` cannot parse JSX and this is the part worth
   testing: the component only applies the number. */

// One control's own size, in board units. Exported: the resize corner is drawn
// at exactly this size too, and the control row has to keep out of it.
export const CONTROL_UNITS = 28;
const CHROME_GAP = 2;          // gap-0.5 between two controls
/* p-px around the row, both sides. It was p-0.5 (4 units) until HQ made the row
   five controls: at five, a nominal-size row is 152 units and the narrowest tile
   a strip actually produces — a portrait picture, half as wide as it is tall,
   160 units at the default pin — has 150.4 units of budget once the "never eat
   the picture" fraction is applied. The row may not be drawn smaller than its
   nominal size (that floor is what keeps the chrome from shrinking to a speck
   when you zoom IN), so the two units had to come from somewhere, and padding
   is the only part of the row that is not a touch target. Sixth control: this
   sum is where you come to decide what it costs — there is nothing left to trim
   here. */
const CHROME_PAD = 2;

/** The WIDTH budget of the control row, in board units, for a given number of
 *  controls — one line, always.
 *
 *  ⚠️ This used to be a constant 64 with a two-column wrap, and the wrap is the
 *  bug that replaced it. Four controls (🔍 ✕ ⬇ 🗑) in two columns is not a
 *  cluster in a corner, it is a 2×2 BLOCK: capped at 70 % of the tile's width
 *  and two rows tall, it landed mid-picture, covered the picture it decorates
 *  and truncated the "step N · strength X" label beside it. A row costs each
 *  target some size at extreme zoom-out (the honest number is in the test), and
 *  it buys back three quarters of the picture at every zoom. */
export function clusterUnits(buttonCount = 4) {
  const n = Math.max(1, Math.floor(Number(buttonCount)) || 1);
  return CONTROL_UNITS * n + CHROME_GAP * (n - 1) + CHROME_PAD;
}

/** The full row: 🔍 ✕ ⬇ HQ 🗑. Five, not four — HQ (show the original file
 *  instead of the fast WebP tile) joined it, and the budget above is exactly
 *  the place that makes the price of a new control visible instead of letting
 *  it be paid silently by every target's size. */
export const CLUSTER_UNITS = clusterUnits(5);
// The share of the node the row is never allowed to exceed. The cap is on the
// ROW, not on one button: capping each button separately let a handful of them
// side by side cover almost the entire width of a small tile. It can afford to
// be near-total now that the row is ONE line drawn along an edge — what a
// control must not do is sit in the middle of the picture, and 0.94 leaves the
// hairline of margin that says "this ends here" without buying a second line.
const MAX_CLUSTER_FRACTION = 0.94;

/**
 * The counter-scale to apply to a pinned node's controls at a given board zoom.
 *
 * 1 means "leave it alone". 2 means "draw it twice as big in board units", which
 * at 50 % zoom is exactly its nominal size on screen.
 *
 * `units` is the row's own width budget (see clusterUnits) and there are TWO
 * ways to reserve the space on the same edge that is not the row's:
 *
 *   • `reserved` — board space drawn at THIS very scale. A node of its own
 *     draws its resize corner with the number this function returns, so the
 *     corner grows and shrinks with the row and both fit in one budget;
 *   • `reservedBoard` — board space already fixed in BOARD units, whatever this
 *     function returns. A group's resize corner is the case: it belongs to the
 *     strip, is counter-scaled by the raw zoom with no cap of its own
 *     (groupCornerScale), and therefore cannot be expressed as a multiple of a
 *     scale it does not use. It is subtracted from the tile before the row is
 *     given what is left.
 */
export function chromeScale(boardScale, nodeW, units = CLUSTER_UNITS, reserved = 0,
  reservedBoard = 0) {
  const s = Number(boardScale);
  const w = Number(nodeW);
  if (!Number.isFinite(s) || s <= 0) return 1;
  const wanted = 1 / s;                       // constant size on screen
  const budget = (Number(units) || CLUSTER_UNITS) + (Number(reserved) || 0);
  const fixed = Number(reservedBoard) || 0;
  const cap = Number.isFinite(w) && w > 0 && budget > 0
    ? Math.max(1, (w * MAX_CLUSTER_FRACTION - fixed) / budget)
    : Infinity;
  return Math.min(Math.max(1, wanted), cap);
}

/** What that control measures on screen, once counter-scaled — the number the
 *  proof is about ("is it big enough for a finger?"). */
export function chromeScreenSize(boardScale, nodeW, units = CLUSTER_UNITS, reserved = 0,
  reservedBoard = 0) {
  const s = Number(boardScale);
  if (!Number.isFinite(s) || s <= 0) return CONTROL_UNITS;
  return CONTROL_UNITS * chromeScale(s, nodeW, units, reserved, reservedBoard) * s;
}

/* ◢ WHERE THE RESIZE CORNERS ARE — one source, because two answers diverged.
 *
 * The control row reserves the corner it sits beside, and that reservation was
 * written as "a member has none, a node of its own has one". Half true, and the
 * half that is false is a bug you can point at: a group MEMBER draws no handle,
 * but the STRIP draws one at its own bottom-right, and the strip's bottom-right
 * corner IS the last member's bottom-right corner. So the last tile of every
 * group had an armed 🗑 sitting on top of the handle that resizes the group.
 *
 * The question the row must ask is therefore not "do I render a handle?" but
 * "is there a handle over this tile?" — and both the rendering and the
 * reservation now read it from here. */

/** Does this tile draw a resize corner of its OWN? (The render condition.) */
export function hasOwnResizeCorner(variant) {
  return variant !== 'member';
}

/** Is there a resize corner over this tile, whoever draws it? (The reservation
 *  condition.) `lastInGroup` is the only member the strip's corner lands on. */
export function hasResizeCornerOver(variant, lastInGroup = false) {
  return hasOwnResizeCorner(variant) || !!lastInGroup;
}

/** The counter-scale a GROUP's resize corner is drawn at (CanvasImageGroup).
 *  Uncapped on purpose — a strip is as wide as it has members, so the cap that
 *  protects a single tile's picture has nothing to protect here. Exported so
 *  the member reserving that corner and the group drawing it cannot drift. */
export function groupCornerScale(boardScale) {
  const s = Number(boardScale);
  return Math.max(1, 1 / Math.max(Number.isFinite(s) ? s : 1, 0.01));
}

/** …and what it therefore measures in BOARD units. */
export function groupCornerUnits(boardScale) {
  return CONTROL_UNITS * groupCornerScale(boardScale);
}

// ✓ The PICK box on a checkpoint pill — same disease as the ✕ above, on the
// control the board's whole generate flow depends on.
//
// It is 12 board units. The board opens on Fit, and Fit on a 400-px phone with a
// few lanes lands around 45 %, where 12 × 0.45 ≈ 5 screen pixels. The toolbar
// tells you to "tick a checkpoint's ✓ to generate from it" and on a phone that
// tick is a five-pixel square. So it is counter-scaled to a CONSTANT size on
// screen, exactly like a pinned image's buttons.
//
// ⚠️ Two bounds, and the second one is why this is not simply 1/scale:
//   • never below 1 — at 100 % and above, the box keeps the size it has always
//     had, so the in-card lineage graph (which passes no scale at all) and every
//     desktop board are untouched;
//   • never wider than a share of the PILL, because the box sits on that pill's
//     top-left corner and a box that outgrows it hides the step number — the
//     exact regression the `left: -6, top: -6` comment in lineageNodes records.
//
// Being honest about what this does NOT achieve: a 40-px finger target inside a
// board zoomed to 45 % is geometrically impossible — 40 screen px is two thirds
// of a whole pill. This makes the box a constant ~12 px instead of a shrinking
// ~5, which is the difference between "fiddly" and "not there". Zooming in
// remains the answer for reliable ticking, and the zoom buttons are now 40 px.
const SELECT_BASE = 12;          // the compact box's own size, in board units
const MAX_SELECT_FRACTION = 0.55;

export function pillSelectScale(boardScale, pillW) {
  const s = Number(boardScale);
  if (!Number.isFinite(s) || s <= 0) return 1;
  const w = Number(pillW);
  const cap = Number.isFinite(w) && w > 0
    ? Math.max(1, (w * MAX_SELECT_FRACTION) / SELECT_BASE)
    : Infinity;
  return Math.min(Math.max(1, 1 / s), cap);
}

/** What that box measures on screen once counter-scaled — the number any proof
 *  about "can a finger hit it?" is actually about. */
export function pillSelectScreenSize(boardScale, pillW) {
  const s = Number(boardScale);
  if (!Number.isFinite(s) || s <= 0) return SELECT_BASE;
  return SELECT_BASE * pillSelectScale(s, pillW) * s;
}

// 🖼🖼 The GRIP of a group of pinned images: the title bar you drag to move the
// whole strip. Same problem as the buttons above and the same answer, with one
// difference — a bar spans the strip's whole width, so it cannot be `scale()`d
// (that would stretch it sideways too). Only its HEIGHT is counter-scaled.
const BAR_BASE = 26;           // its height on screen, in pixels
const MAX_BAR_FRACTION = 0.35; // …and the share of the strip it may never exceed

/**
 * How tall a group's drag bar must be drawn, in BOARD units, so it stays a
 * finger-sized grip whatever the zoom.
 *
 * This bar is the ONLY way to move a group — dragging a picture inside a group
 * means "take this one out" — so a bar that shrinks to four pixels at 24 % does
 * not make the gesture awkward, it makes the group immovable.
 */
export function groupBarHeight(boardScale, groupH) {
  const s = Number(boardScale);
  if (!Number.isFinite(s) || s <= 0) return BAR_BASE;
  return Math.min(Math.max(BAR_BASE, BAR_BASE / s), groupBarMaxHeight(groupH));
}

// 🛝 The grab band along a LANE's bottom edge — the grip that sets how much
// room that dataset keeps. Same disease as the bar above: a 6-px edge at 24 %
// zoom is a 1.4-px target, which is not a hard grip, it is no grip.
const LANE_EDGE_BASE = 8;      // its thickness on screen, in pixels
const MAX_EDGE_FRACTION = 0.2; // …and the share of the lane it may never exceed

/**
 * How thick a lane's bottom-edge resize grip must be drawn, in BOARD units.
 *
 * Capped as a fraction of the lane, for the reason the bar's cap exists: a
 * short lane whose grab band was a third of its height would be a lane you
 * cannot click INTO — the band would sit over its own content.
 */
export function laneEdgeHeight(boardScale, laneH) {
  const s = Number(boardScale);
  const h = Number(laneH);
  const cap = Number.isFinite(h) && h > 0
    ? Math.max(LANE_EDGE_BASE, h * MAX_EDGE_FRACTION) : LANE_EDGE_BASE;
  if (!Number.isFinite(s) || s <= 0) return LANE_EDGE_BASE;
  return Math.min(Math.max(LANE_EDGE_BASE, LANE_EDGE_BASE / s), cap);
}

/**
 * The TALLEST that bar can ever get on this strip, in board units — its height
 * at maximum zoom-out, where the counter-scale saturates against the cap.
 *
 * Why this exists as its own number: the bar is drawn ABOVE the strip's box, so
 * it occupies board space that belongs to no node. Anything the board places
 * there lands on top of the group's only grip. The placers therefore have to
 * treat a group as taller than it looks, and they cannot ask "how tall at the
 * current zoom?" — a picture placed at 100 % must still not be under the bar
 * once the user zooms out to 40 %, where the bar is TWICE as tall (26 board
 * units becomes 52.5 on a 150-unit strip). So they reserve the worst case,
 * which is exactly the cap and does not depend on the scale at all.
 */
export function groupBarMaxHeight(groupH) {
  const h = Number(groupH);
  return Number.isFinite(h) && h > 0 ? Math.max(BAR_BASE, h * MAX_BAR_FRACTION) : BAR_BASE;
}

/**
 * Is this event target one of a pinned node's own BUTTONS (✕, 🔍)?
 *
 * THE guard, extracted out of the pointer handler so it cannot be lost the next
 * time that handler is rewritten. A pointerdown here must not be captured by the
 * frame and must not arm the long press: a captured pointer retargets the click
 * that follows to the frame, and the button never hears it — which is a second,
 * quieter way for a ✕ to "do nothing", on every pointer type this time.
 *
 * The RESIZE corner is deliberately not one of these. It has no click of its
 * own — it is a gesture — so it WANTS the capture, and only its long-press
 * exemption sets it apart from the picture. See nodePointerIntent.
 *
 * Takes anything with `closest` (a DOM node, or a stub in a test).
 */
export function isNodeControlTarget(target) {
  if (!target || typeof target.closest !== 'function') return false;
  // A group's own ✕ is on the strip, not inside any one picture, so it needs
  // naming here too — otherwise the frame captures its pointer and the button
  // never hears the click, which is exactly the bug this guard exists for.
  //
  // `data-canvas-control` is the OPEN version of the same rule, for a control
  // that lives in the zoomed world without belonging to a pinned node — the
  // lane header's 🪪 reference thumbnail was the first. It was written, it was
  // correct, and a real click on it did nothing at all: the frame captured the
  // pointer and the click that followed was retargeted away from the button.
  // Anything added to the world from now on can opt out by wearing this
  // attribute instead of rediscovering the trap.
  return !!(target.closest('[data-canvas-image] button')
    || target.closest('[data-canvas-group-bar] button')
    || target.closest('[data-canvas-control]'));
}

/**
 * What a pointerdown inside a pinned node means:
 *   'control'    — a button: hands off entirely, no capture, no long press;
 *   'group-move' — a group's title bar: move the WHOLE strip, on any pointer
 *                  type. It is a bar you deliberately grabbed; making a finger
 *                  wait out a long press on it would be gratuitous, and it is
 *                  the only grip a group has;
 *   'resize'     — the corner handle: resize immediately, on every pointer type;
 *   'move'       — mouse/pen on the picture: pick it up;
 *   'press'      — touch on the picture: pan for now, pick it up on a long press.
 *
 * One function instead of a chain of `if`s inside the handler, so the rule
 * "a control is never a gesture" is a thing a test can hold on to.
 */
export function nodePointerIntent(target, pointerType) {
  if (!target || typeof target.closest !== 'function') return 'press';
  if (target.closest('[data-canvas-image] button')
    || target.closest('[data-canvas-group-bar] button')
    || target.closest('[data-canvas-control]')) return 'control';
  if (target.closest('[data-canvas-group-bar]')) return 'group-move';
  if (target.closest('[data-canvas-image-resize]')) return 'resize';
  return pointerType === 'touch' ? 'press' : 'move';
}
