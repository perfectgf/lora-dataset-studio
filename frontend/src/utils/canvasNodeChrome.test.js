import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CLUSTER_UNITS, CONTROL_UNITS, chromeScale, chromeScreenSize, clusterUnits,
  groupBarHeight, isNodeControlTarget, nodePointerIntent, pillSelectScale,
  pillSelectScreenSize,
} from './canvasNodeChrome.js';

/* The ✕ that "did not work" on a phone.

   The first test is the bug, in one number: at the zoom the report came from,
   the close button measured about ten screen pixels. Nothing was broken —
   nothing was reachable. */

// A stand-in for a DOM node: `closest` answers from a list of selectors the
// element is inside of. Enough to pin the rules without a DOM.
const el = (...inside) => ({
  closest: (sel) => (inside.includes(sel) ? { sel } : null),
});

const CLOSE_BUTTON = el('[data-canvas-image] button', '[data-canvas-image]');
const RESIZE_CORNER = el('[data-canvas-image-resize]', '[data-canvas-image]');
const PICTURE = el('[data-canvas-image]');

// ---- THE bug --------------------------------------------------------------

test('the ✕ is a real touch target at the zoom the board is actually read at', () => {
  // 65 % — the zoom on the report. Untouched, the 28-unit button lands at 18 px.
  assert.ok(28 * 0.65 < 24, 'the raw control is under any touch guideline');
  // Counter-scaled, it is back to its nominal size under the finger.
  assert.ok(chromeScreenSize(0.65, 320) >= 24,
    `got ${chromeScreenSize(0.65, 320)} px at 65 %`);
  // And at the far end of the range — a board fitted to twenty runs. Here the
  // row's cap bites before the ideal does (a 320-unit pin is only 77 px on
  // screen at that zoom, and four 28-px buttons would be twice as wide as the
  // whole tile), so the honest claim is "about three times bigger than the
  // bug", not "28 px".
  assert.ok(chromeScreenSize(0.24, 320) >= 16,
    `got ${chromeScreenSize(0.24, 320)} px at 24 %`);
  assert.ok(chromeScreenSize(0.24, 320) > 3 * (14 * 0.24),
    'and far bigger than the glyph it replaces');
});

test('zooming IN does not blow the control up', () => {
  assert.equal(chromeScale(1, 320), 1);
  assert.equal(chromeScale(2.5, 320), 1, 'never smaller than its board size');
});

test('the control CLUSTER never eats the picture it decorates', () => {
  // A contact-sheet thumbnail on a board fitted very far out: a constant-size
  // pair of buttons would cover the whole tile. The cap wins, and the fact that
  // the target is then below the touch guideline is a consequence of the tile
  // being 45 px wide — there is no honest way around that, and it is stated
  // here rather than hidden. Zooming in is the answer, and it works.
  const tiny = chromeScale(0.28, 160);
  assert.ok(tiny * CLUSTER_UNITS <= 160 * 0.94 + 1e-9,
    'the 🔍 ✕ ⬇ 🗑 row stays inside the tile');
  assert.ok(tiny < 1 / 0.28, 'the cap, not the ideal, is what applies');
  // A full-size pin at the same zoom has room for more, and gets it.
  assert.ok(chromeScreenSize(0.28, 320) > chromeScreenSize(0.28, 160));
});

test('the controls are ONE line, and each one is paid for in width', () => {
  // The bug this pins is the WRAP, not the size. Four controls (🔍 ✕ ⬇ 🗑) laid
  // out in two columns is a 2×2 block, and a block capped at a share of the
  // tile's width is a block sitting mid-picture: reported on a group of five,
  // it covered the picture AND the "step N · strength X" label beside it.
  // A row's budget therefore GROWS with the number of controls — that is the
  // honest cost, and it is a number rather than a wrap.
  assert.equal(CLUSTER_UNITS, clusterUnits(4));
  assert.equal(clusterUnits(4), 4 * CONTROL_UNITS + 3 * 2 + 4,
    'four targets, three gaps, one padding — one line');
  assert.ok(clusterUnits(4) > clusterUnits(3), 'a fourth control widens the row');
  assert.ok(clusterUnits(3) > clusterUnits(2));
  // What the row still guarantees at the zooms the board is actually read at.
  assert.ok(chromeScreenSize(0.45, 320) >= 24,
    `got ${chromeScreenSize(0.45, 320)} px at 45 %`);
  // …and what it costs at the far end, stated instead of hidden: a four-button
  // row on a 77-px-wide tile cannot give 20 px per target — 4 × 20 is twice the
  // tile. It gives ~16, against ~20 for the block that covered the picture.
  const far = chromeScreenSize(0.24, 320);
  assert.ok(far >= 16 && far < 20, `got ${far} px at 24 %`);
});

test('the row leaves the resize corner alone', () => {
  // The corner is drawn on the same edge, at the same counter-scale. Reserved,
  // row + corner still fit the tile; unreserved, the row would grow into it
  // exactly when the cap starts to bite (i.e. on the smallest tiles).
  const w = 260;
  const k = chromeScale(0.3, w, clusterUnits(4), CONTROL_UNITS);
  assert.ok(k * (clusterUnits(4) + CONTROL_UNITS) <= w * 0.94 + 1e-9,
    'the row stops where the resize handle starts');
  assert.ok(k < chromeScale(0.3, w, clusterUnits(4)),
    'reserving the corner is what costs that width — a member pays nothing');
  // A member has no corner to reserve, and asking for fewer controls (no 🗑
  // wired) must not make the remaining ones smaller.
  assert.ok(chromeScale(0.3, w, clusterUnits(3)) >= chromeScale(0.3, w, clusterUnits(4)));
});

test('a nonsense zoom is not allowed to produce a nonsense control', () => {
  assert.equal(chromeScale(0, 320), 1);
  assert.equal(chromeScale(-3, 320), 1);
  assert.equal(chromeScale(NaN, 320), 1);
  assert.equal(chromeScale(0.5, NaN), 2, 'no node width ⇒ no cap, just the ideal');
});

// ---- the guard: a control is never a gesture ------------------------------

test('a press on the ✕ is a control, never a drag and never a long press', () => {
  assert.equal(nodePointerIntent(CLOSE_BUTTON, 'touch'), 'control');
  assert.equal(nodePointerIntent(CLOSE_BUTTON, 'mouse'), 'control');
  assert.ok(isNodeControlTarget(CLOSE_BUTTON));
});

test('the resize corner answers at once, finger or mouse', () => {
  assert.equal(nodePointerIntent(RESIZE_CORNER, 'touch'), 'resize');
  assert.equal(nodePointerIntent(RESIZE_CORNER, 'mouse'), 'resize');
  assert.equal(isNodeControlTarget(RESIZE_CORNER), false,
    'the corner is a GESTURE, not a button: it wants the pointer capture');
});

test('the picture itself still moves — the fix must not cost the drag', () => {
  assert.equal(nodePointerIntent(PICTURE, 'mouse'), 'move');
  assert.equal(nodePointerIntent(PICTURE, 'pen'), 'move');
  assert.equal(nodePointerIntent(PICTURE, 'touch'), 'press',
    'touch pans until the long press picks the node up');
  assert.ok(!isNodeControlTarget(PICTURE));
});

test('something with no closest() at all does not throw', () => {
  assert.equal(isNodeControlTarget(null), false);
  assert.equal(isNodeControlTarget({}), false);
  assert.equal(nodePointerIntent(null, 'touch'), 'press');
});

// ---- 🖼🖼 the grip of a GROUP of pinned images ----------------------------

const GROUP_BAR = el('[data-canvas-group-bar]', '[data-canvas-group]');
const GROUP_CLOSE = el('[data-canvas-group-bar] button', '[data-canvas-group-bar]',
  '[data-canvas-group]');

test('a group’s title bar moves the whole strip, on any pointer type', () => {
  assert.equal(nodePointerIntent(GROUP_BAR, 'mouse'), 'group-move');
  assert.equal(nodePointerIntent(GROUP_BAR, 'touch'), 'group-move',
    'the bar is the only grip a group has — a finger must not have to wait');
  assert.equal(isNodeControlTarget(GROUP_BAR), false, 'it is a gesture: it wants the capture');
});

test('a group’s own ✕ is a button, exactly like a picture’s', () => {
  assert.equal(nodePointerIntent(GROUP_CLOSE, 'touch'), 'control');
  assert.ok(isNodeControlTarget(GROUP_CLOSE),
    'without this the frame captures the pointer and the ✕ never hears the click');
});

test('the group bar stays a finger-sized grip at the zoom the board is read at', () => {
  // 400 board units tall, read at 24 % — the far end of "zoomed out".
  const h = groupBarHeight(0.24, 400);
  assert.ok(h * 0.24 >= 24, `the bar measured ${(h * 0.24).toFixed(1)} px on screen`);
  // Zoomed IN it must not balloon: a constant screen size, never smaller.
  assert.equal(groupBarHeight(2, 400), 26);
});

test('the group bar never eats the strip it labels', () => {
  // A tiny strip at a tiny zoom: the counter-scale is capped by the strip.
  assert.ok(groupBarHeight(0.05, 100) <= Math.max(26, 100 * 0.35) + 1e-9);
});

test('groupBarHeight survives nonsense', () => {
  assert.equal(groupBarHeight(0, 400), 26);
  assert.equal(groupBarHeight(NaN, 400), 26);
  assert.ok(Number.isFinite(groupBarHeight(1, NaN)));
});

/* 🪪 A control that lives in the zoomed world WITHOUT belonging to a pinned
   node — the lane header's reference thumbnail is the first one. Caught in the
   browser, not by a test: the button was correct, its handler was correct, and
   clicking it did nothing, because the frame captured the pointer and the click
   that followed was retargeted away from it. The generic marker exists so the
   next thing added to the world opts out instead of rediscovering this. */
const LANE_CONTROL = el('[data-canvas-control]');

test('a control in the world opts out of the board gesture by marker alone', () => {
  assert.ok(isNodeControlTarget(LANE_CONTROL),
    'no capture ⇒ the click reaches the button');
  assert.equal(nodePointerIntent(LANE_CONTROL, 'mouse'), 'control');
  assert.equal(nodePointerIntent(LANE_CONTROL, 'touch'), 'control',
    'and it never arms the long press either');
});

/* ✓ The pick box on a checkpoint pill — the same disease, on the control the
   board's whole generate flow runs through. Measured headless at 400 px: the
   board opens on Fit, Fit landed at 45 %, and the box reported 5 × 5 CSS px
   while the toolbar told the user to tick it. */

test('the ✓ pick box stops shrinking with the board', () => {
  // The bug, in one number.
  assert.ok(12 * 0.45 < 6, 'raw, the box is a five-pixel square at Fit zoom');
  // Counter-scaled it holds its nominal size on screen instead.
  assert.ok(pillSelectScreenSize(0.45, 60) > 10,
    'more than twice the size it had, at the zoom a phone opens on');
  assert.ok(Math.abs(pillSelectScreenSize(0.45, 200) - 12) < 1e-9,
    'on a wide pill nothing caps it: exactly its nominal 12 px');
});

test('the ✓ box never grows over the pill it sits on', () => {
  // A 60-unit pill shows a four-digit step. The box may take 55 % of it and no
  // more — larger and the leading digit goes, which is a fix already paid for.
  assert.ok(12 * pillSelectScale(0.1, 60) <= 60 * 0.55 + 1e-9);
  assert.ok(12 * pillSelectScale(0.01, 40) <= 40 * 0.55 + 1e-9);
});

test('the ✓ box is untouched at 100 % and above — and without a board at all', () => {
  // The in-card lineage graph passes no scale; a desktop board reads at 100 %+.
  assert.equal(pillSelectScale(1, 60), 1);
  assert.equal(pillSelectScale(2.5, 60), 1);
  assert.equal(pillSelectScale(0, 60), 1);
  assert.equal(pillSelectScale(NaN, 60), 1);
  assert.ok(Number.isFinite(pillSelectScale(0.4, NaN)));
});
