/* 📐 The markers the responsive probe measures the Datasets workspace by.
 *
 * Same contract as bankProbeMarkers.test.js: `scripts/responsiveProbe.mjs`
 * finds its surfaces by attribute, and these assertions keep the attributes
 * in place. Source text cannot say the layout is good; it can say the probe
 * is still pointed at the right elements.
 *
 * First measured 2026-08-23: 224 violations on the first run (208 controls
 * under 40 px, the two chip rails overlapping by 4 px, a closed ⋯ More menu
 * billed to the fold), 0 after. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8');
const workspace = read('./DatasetWorkspace.jsx');
const lightbox = read('./DatasetLightbox.jsx');
const probe = read('../../../scripts/responsiveProbe.mjs');

test('every fixed surface of the workspace is marked for the responsive probe', () => {
  for (const surface of ['header', 'more-menu', 'sections', 'destinations', 'grid-toolbar', 'filter-bar']) {
    assert.ok(workspace.includes(`data-probe-chrome="${surface}"`),
      `the ${surface} lost its data-probe-chrome marker — the probe stops measuring it`);
  }
  // the desktop rail is a side column: a panel (fill), never chrome (a column
  // is not a share of the fold)
  assert.match(workspace, /data-probe-panel="sections-rail"/);
  assert.match(workspace, /data-probe-chrome="more-menu" data-probe-panel="more-menu"/);
});

test('the lightbox is a layer', () => {
  assert.match(lightbox, /data-probe-chrome="lightbox" data-probe-layer/);
});

test('the two chip rails do not overlap, and the second one folds on a phone held sideways', () => {
  // -mt-1 on the destinations rail put it 4 px under the sections rail
  assert.ok(!/destinations.*\n.*-mt-1/.test(workspace), 'the destinations rail pulls itself under the sections rail again');
  assert.match(workspace, /data-probe-chrome="destinations"\n\s+className="relative -mx-4 overflow-x-auto px-4 pb-1 lg:hidden \[@media\(max-height:500px\)\]:hidden"/);
});

test('controls reach 40 px below lg — the section chips, the header, the menu, the toolbar chips', () => {
  assert.match(workspace, /\? `flex min-h-10 shrink-0 items-center gap-1\.5 whitespace-nowrap rounded-full border px-3 py-1\.5 text-xs font-medium/);
  assert.match(workspace, /\? `min-h-10 shrink-0 whitespace-nowrap rounded-full border px-3 py-1\.5 text-xs/);
  assert.match(workspace, /const MENU_ITEM = 'min-h-10 lg:min-h-0 w-full flex items-center/);
  assert.match(workspace, /className=\{`min-h-10 lg:min-h-0 px-2 py-0\.5 rounded-full border text-\[0\.6875rem\] font-semibold tabular-nums/);
  assert.match(lightbox, /min-h-10 lg:min-h-9/);
});

test('the probe opens the dataset first, and names the Images button exactly', () => {
  assert.match(probe, /'#\/datasets': \{/);
  assert.match(probe, /prime: \['\[aria-label\^="Open the dataset"\]'\]/);
  // `text=Images` is a substring match that lands on "Add images" first
  assert.match(probe, /button:has-text\("Images"\):not\(:has-text\("Add"\)\)/);
});
