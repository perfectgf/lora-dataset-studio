import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DOUBLE_TAP_ZOOM, FIT_VIEW, MAX_MAX_ZOOM, MIN_MAX_ZOOM, MIN_ZOOM, TAP_MAX_MS, TAP_SLOP_PX,
  clampPan, clampZoom, doubleTapView, fitSize, isZoomed, maxZoomFor, panByDelta,
  tapOutcome, zoomAtPoint,
} from './imageZoomPan.js';

/* A 4:3 render on a 412-px phone: `object-contain` draws it 412x309 in a
   412x780 frame. That is the exact case this module was written for — the one
   shape where folding the details away gains nothing, because the picture
   already has the whole of the scarce axis. */
const PHONE = { fitW: 412, fitH: 309, frameW: 412, frameH: 780 };

test('fit is where a picture opens, and where every way back leads', () => {
  assert.deepEqual({ ...FIT_VIEW }, { scale: 1, tx: 0, ty: 0 });
  assert.equal(isZoomed(FIT_VIEW), false);
  assert.equal(isZoomed({ scale: 1.0000001, tx: 0, ty: 0 }), false, 'float noise is not a zoom');
  assert.equal(isZoomed({ scale: 1.4 }), true);
});

/* Rule 2. Past the file's own pixels you are magnifying interpolation, and the
   whole reason to zoom a render is to judge whether the model got an eye
   right — a viewer that offers 20x on a 512-px file is lying about what it can
   show you. */
test('zoom stops where the file’s detail stops, with a floor and a ceiling', () => {
  // 1024-px file drawn 412 wide: 2.49 real pixels of headroom.
  assert.ok(Math.abs(maxZoomFor(1024, 412) - 1024 / 412) < 1e-9);
  // A picture drawn LARGER than its file still gets a usable magnifier rather
  // than "you may not zoom", which is never the right answer from one.
  assert.equal(maxZoomFor(300, 412), MIN_MAX_ZOOM);
  // …and a 6000-px render does not offer a 15x nobody can aim on a phone.
  assert.equal(maxZoomFor(6000, 412), MAX_MAX_ZOOM);
  // Unmeasurable (not laid out, broken file) falls back to something usable.
  assert.equal(maxZoomFor(0, 412), MIN_MAX_ZOOM);
  assert.equal(maxZoomFor(1024, 0), MIN_MAX_ZOOM);
  // The clamp obeys the cap it is handed, both ends.
  assert.equal(clampZoom(99, 2.5), 2.5);
  assert.equal(clampZoom(0.2, 2.5), MIN_ZOOM);
  // A cap under 1 cannot forbid fit — that would strand the picture nowhere.
  assert.equal(clampZoom(1, 0.4), MIN_ZOOM);
});

/* Rule 1, the one every hand-written pan gets wrong first: one careless swipe
   flings the picture off screen and the only way back is a reload. */
test('the picture can never be panned away from the frame', () => {
  // At fit the picture is NARROWER than the frame is tall, so there is no
  // travel at all — and asking for some re-centres it rather than moving it.
  const centred = panByDelta({ scale: 1, tx: 0, ty: 0 }, 500, -900, PHONE);
  assert.deepEqual(centred, { scale: 1, tx: 0, ty: 0 });

  // At 3x the picture is 1236x927 in a 412x780 frame: 824 px of horizontal
  // overflow and 147 of vertical, so exactly 412 and 73.5 px of travel each way.
  const z = { scale: 3, tx: 0, ty: 0 };
  assert.equal(panByDelta(z, 10_000, 0, PHONE).tx, 412);
  assert.equal(panByDelta(z, -10_000, 0, PHONE).tx, -412);
  assert.equal(panByDelta(z, 0, 10_000, PHONE).ty, 73.5);
  assert.equal(panByDelta(z, 0, -10_000, PHONE).ty, -73.5);
  // A move inside the travel is passed through untouched.
  assert.deepEqual(panByDelta(z, 40, -20, PHONE), { scale: 3, tx: 40, ty: -20 });

  // Zooming back out drags the picture home with it: the travel shrinks to
  // zero, so a view that was parked at the edge cannot stay there.
  assert.deepEqual(clampPan({ scale: 1, tx: 412, ty: 73.5 }, PHONE), { scale: 1, tx: 0, ty: 0 });

  // An unmeasurable box does not throw and does not invent travel.
  assert.deepEqual(clampPan({ scale: 2, tx: 50, ty: 50 }, {}), { scale: 2, tx: 0, ty: 0 });
});

/* Anchoring is the whole difference between a magnifier and a control that
   throws away what you were looking at. Pinch on a face and the face is what
   gets bigger — not the middle of the picture. */
test('zoom keeps what is under the fingers under the fingers', () => {
  const box = { fitW: 400, fitH: 400, frameW: 400, frameH: 400 };
  // A point 100 px right of centre, doubled: it must still be 100 px right of
  // centre afterwards. Solving p - k(p - t) with p=100, t=0, k=2 gives -100 —
  // the picture slides left by exactly the amount that keeps the point still.
  const out = zoomAtPoint({ scale: 1, tx: 0, ty: 0 }, 2, { x: 100, y: 0 }, box, 8);
  assert.equal(out.scale, 2);
  assert.equal(out.tx, -100);
  // …and the invariant itself, stated rather than trusted: the image-space
  // offset under the point does not move.
  const before = (100 - 0) / 1;
  const after = (100 - out.tx) / out.scale;
  assert.ok(Math.abs(before - after) < 1e-9);
  // Anchored at the centre nothing slides.
  assert.deepEqual(zoomAtPoint({ scale: 1, tx: 0, ty: 0 }, 2, { x: 0, y: 0 }, box, 8),
    { scale: 2, tx: 0, ty: 0 });
  // The result is clamped like any other view — a zoom cannot leave a gap.
  const edge = zoomAtPoint({ scale: 1, tx: 0, ty: 0 }, 1.2, { x: 10_000, y: 0 }, box, 8);
  assert.equal(edge.tx, -(400 * 1.2 - 400) / 2);
  // The cap applies to the anchored zoom too.
  assert.equal(zoomAtPoint({ scale: 1 }, 100, { x: 0, y: 0 }, box, 2.5).scale, 2.5);
});

/* A gesture whose job is "put it back" has to put it back in one go, from any
   zoom, or the way home is a number of taps the user has to count. */
test('double tap goes in from fit, and all the way out from anywhere else', () => {
  const box = { fitW: 400, fitH: 300, frameW: 400, frameH: 800 };
  const inward = doubleTapView(FIT_VIEW, { x: 0, y: 0 }, box, 8);
  assert.equal(inward.scale, DOUBLE_TAP_ZOOM);
  // From anything above fit — one notch or the maximum — it lands exactly home.
  for (const scale of [1.05, 2, 8]) {
    assert.deepEqual(doubleTapView({ scale, tx: 120, ty: -40 }, { x: 30, y: 30 }, box, 8),
      { scale: 1, tx: 0, ty: 0 });
  }
  // It goes in ANCHORED, like every other zoom here.
  const off = doubleTapView(FIT_VIEW, { x: 80, y: 0 }, box, 8);
  assert.ok(off.tx < 0, 'zooming in at a point right of centre slides the picture left');
  // …and it never exceeds what this file can actually show.
  assert.equal(doubleTapView(FIT_VIEW, { x: 0, y: 0 }, box, 2).scale, 2);
});

/* The one piece of geometry the browser knows and never tells you: what size
   `object-contain` settled on. Every rule above is expressed in those pixels. */
test('the drawn size of a contained picture is derived, not guessed', () => {
  // 4:3 into a tall phone frame — width-bound, which is the whole reason this
  // module exists.
  assert.deepEqual(fitSize(1024, 768, 412, 780), { width: 412, height: 309 });
  // …and height-bound in a wide one.
  assert.deepEqual(fitSize(1000, 1000, 800, 400), { width: 400, height: 400 });
  // Not laid out yet is reported as zero, never as one pixel.
  assert.deepEqual(fitSize(0, 768, 412, 780), { width: 0, height: 0 });
  assert.deepEqual(fitSize(1024, 768, 412, 0), { width: 0, height: 0 });
});

/* ⚠️ THE ONE THAT WAS MEASURED WRONG FIRST.
   Driven in a real headless browser at 412x780 on a real render: pinch out,
   let go — and the details folded away every single time. The surviving finger
   of a pinch is handed back as an ordinary press, and letting go of it ends a
   press that has moved 0 px in 0 ms, which is byte-for-byte the shape of a tap.
   The browser found it; this keeps it found. */
test('the last finger of a pinch is finishing a pinch, not tapping', () => {
  const tapish = { moved: 0, held: 0, onImage: true };
  assert.equal(tapOutcome(tapish, { pendingTap: false }), 'single');
  assert.equal(tapOutcome({ ...tapish, fromPinch: true }, { pendingTap: false }), 'ignore');
  // …and it cannot complete somebody else's double tap either.
  assert.equal(tapOutcome({ ...tapish, fromPinch: true }, { pendingTap: true }), 'ignore');
});

test('a press only becomes a tap when it was still, short, and on the picture', () => {
  const ok = { moved: 0, held: 100, onImage: true };
  assert.equal(tapOutcome(ok, {}), 'single');
  assert.equal(tapOutcome(ok, { pendingTap: true }), 'double');
  // A thumb wobbles; a drag does not.
  assert.equal(tapOutcome({ ...ok, moved: TAP_SLOP_PX }, {}), 'single');
  assert.equal(tapOutcome({ ...ok, moved: TAP_SLOP_PX + 0.5 }, {}), 'ignore');
  // A hold is not a tap.
  assert.equal(tapOutcome({ ...ok, held: TAP_MAX_MS }, {}), 'single');
  assert.equal(tapOutcome({ ...ok, held: TAP_MAX_MS + 1 }, {}), 'ignore');
  // The backdrop belongs to the viewer's close, not to this gesture.
  assert.equal(tapOutcome({ ...ok, onImage: false }, {}), 'ignore');
  // Nothing at all is not a tap.
  assert.equal(tapOutcome(null, {}), 'ignore');
  assert.equal(tapOutcome(undefined, { pendingTap: true }), 'ignore');
});
