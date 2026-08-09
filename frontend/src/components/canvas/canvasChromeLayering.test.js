/**
 * The board's polish pass, in the two places it can be held to account without
 * a browser: the STACKING contract between the filter panel and the board, and
 * the low-zoom legibility arithmetic.
 *
 * The first is a source test on class names, deliberately. It is not testing
 * CSS — it is testing that two specific promises are still WRITTEN down, because
 * both are one class name away from silently disappearing in a rewrite and the
 * symptom (a pinned strip covering the only control that can take a lane off
 * the board) reads as a layout bug nobody can locate.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  isLowZoom, LOW_ZOOM_THRESHOLD, MIN_LABEL_ZOOM, showsZoomLabels,
  zoomLabelScale, zoomLabelText,
} from '../../utils/canvasZoomLegibility.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const read = (f) => readFileSync(join(HERE, f), 'utf8');

/* ------------------------------------------------------- stacking contract */

test('the board frame is an isolated stacking context', () => {
  // Without it, "the board cannot cover the filter panel" rests on a single
  // `overflow-hidden`, and every z-index inside the frame is competing in the
  // PAGE's stacking context.
  const src = read('LineageCanvas.jsx');
  const cls = /className="lds-canvas-frame ([^"]+)"/.exec(src);
  assert.ok(cls, 'the frame must keep its lds-canvas-frame handle');
  assert.match(cls[1], /\bisolate\b/);
  assert.match(cls[1], /\boverflow-hidden\b/);
});

test('the dataset filter sits above the board and owns its own context', () => {
  const src = read('CanvasDatasetFilter.jsx');
  const cls = /className="lds-canvas-filter ([^"]+)"/.exec(src);
  assert.ok(cls, 'the filter must keep its lds-canvas-filter handle');
  assert.match(cls[1], /\brelative\b/);
  assert.match(cls[1], /\bisolate\b/);
  assert.match(cls[1], /\bz-20\b/);
});

test('a floating panel is painted with an OPAQUE token', () => {
  // `bg-surface` and `bg-surface-overlay` read as synonyms and are not: the
  // first is an alpha-baked tint (4 % white, measured) for lifting a card off
  // the page. A popover painted with it is a sheet of glass — the toolbar and
  // the board are legible straight through the open menu. Nothing that FLOATS
  // may use it.
  // Only the FLOATING container is in scope: a button drawn inside the popover
  // is on top of an opaque panel already, and `bg-surface` is the right tint
  // for it. The container is the line that carries both `absolute` and a
  // shadow — that pairing is what "floats" means here.
  for (const f of ['CanvasFilterMenu.jsx', 'CanvasLayoutPresets.jsx']) {
    // Class strings here are concatenated across source lines, so the pieces
    // are glued back together first: asserting on a line would be asserting on
    // where the author happened to wrap, not on what the browser receives.
    const joined = read(f).replace(/'\s*\n\s*\+\s*'/g, '');
    const floating = joined.split('\n')
      .filter((l) => /\babsolute\b/.test(l) && /shadow-/.test(l));
    assert.ok(floating.length >= 1, `${f}: no floating panel found`);
    for (const line of floating) {
      assert.match(line, /\bbg-surface-overlay\b/, `${f} floats on a transparent panel`);
    }
  }
});

test('the board’s top overlay does not CLIP the filter’s menus', () => {
  // It was `overflow-y-auto`, which was right while the filter was a tall
  // fold-out panel. Against popovers a scroll container is a guillotine: the
  // Datasets menu opened 354 px tall inside a 76-px box and showed a 20-px
  // sliver. Measured on the real page, not reasoned about.
  const src = read('LineageCanvas.jsx');
  const top = /className="pointer-events-none absolute inset-x-0 top-0 z-20[^"]*"/.exec(src);
  assert.ok(top, 'the top overlay must keep its handle');
  assert.match(top[0], /overflow-visible/);
  assert.doesNotMatch(top[0], /overflow-y-auto/);
});

test('the filter costs the board a row of chips, not a panel', () => {
  const src = read('CanvasDatasetFilter.jsx');
  // No fold-out body at all any more: the controls live in popovers, so there
  // is nothing that can grow between the page title and the board.
  assert.doesNotMatch(src, /max-h-\[\d+vh\]/);
  assert.match(src, /flex flex-wrap items-center gap-1\.5/);
  // The dataset list scrolls inside its own popover instead.
  assert.match(src, /max-h-64[^"]*overflow-y-auto/);
});

/* -------------------------------------------------------- low-zoom labels */

test('labels appear only where the card’s own text has stopped being readable', () => {
  assert.equal(isLowZoom(1), false);
  assert.equal(isLowZoom(LOW_ZOOM_THRESHOLD), false);
  assert.equal(isLowZoom(0.36), true);
  // 36 % — the zoom a real board of a dozen lanes actually opens at, and
  // squarely inside the band this exists for.
  assert.equal(showsZoomLabels(0.36), true);
  assert.equal(showsZoomLabels(1), false);
});

test('below the floor nothing is drawn rather than a row of overlapping blobs', () => {
  assert.equal(showsZoomLabels(MIN_LABEL_ZOOM / 2), false);
  assert.equal(showsZoomLabels(0), false);
  assert.equal(showsZoomLabels(NaN), false);
});

test('the label is counter-scaled but never outgrows the card it labels', () => {
  // At 36 % a constant size on screen means ×2.78 — well inside the cap.
  assert.ok(Math.abs(zoomLabelScale(0.36, 264) - 1 / 0.36) < 1e-9);
  // At 10 %, 1/scale is ×10 and a 64-unit badge would be 640 units wide on a
  // 264-unit card. The cap wins.
  const capped = zoomLabelScale(0.1, 264);
  assert.ok(capped < 10, 'the cap must bite');
  assert.ok(capped * 64 <= 264 * 0.9 + 1e-9, 'never wider than its share of the card');
  // Zoomed IN, the badge is not drawn at all — and if it were, it would never
  // be shrunk below its own size.
  assert.equal(zoomLabelScale(2, 264), 1);
});

test('when the lane header has gone unreadable too, the badge says whose run it is', () => {
  assert.equal(zoomLabelText({ record_id: 81 }, 'Ada', 0.4), '#81');
  assert.equal(zoomLabelText({ record_id: 81 }, 'Ada', 0.2), 'Ada #81');
  assert.equal(zoomLabelText({}, 'Ada', 0.4), 'run');
});
