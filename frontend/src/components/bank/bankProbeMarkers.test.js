/* 📐 The markers the responsive probe measures the Bank by.
 *
 * `scripts/responsiveProbe.mjs` renders the Bank workspace at real viewport
 * sizes and measures what came out. It finds the surfaces by attribute, never
 * by class — which leaves one hole: delete the attributes and it measures
 * nothing and says the page is fine. These assertions run in `node --test`, on
 * every commit, and keep the probe pointed at the right elements.
 *
 * ⚠️ Source text cannot tell you the layout is good; only the probe can. This
 * file guarantees that the thing which CAN tell you still has its markers.
 *
 * First measured 2026-08-23 on a throwaway instance: 412 violations on the
 * first run (a 1 550-px passes panel opening itself on a 360-px phone, 384
 * controls under 40 px, a header taking 38 % of the fold at rest), 0 after. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8');
const workspace = read('./BankWorkspace.jsx');
const rail = read('./BankFilterRail.jsx');
const lightbox = read('./BankReviewLightbox.jsx');
const passes = read('./BankPassesPanel.jsx');
const layout = read('./bankLayout.js');
const page = read('../../pages/BankPage.jsx');
const probe = read('../../../scripts/responsiveProbe.mjs');

test('the Bank header and its sheets are marked for the responsive probe', () => {
  assert.match(workspace, /<header data-probe-chrome="header"/);
  for (const sheet of ['auto-reject', 'curate-diverse', 'curate-balanced', 'curate-text']) {
    // A sheet sits over the grid behind a click-to-close backdrop: a layer (not
    // budgeted, paired with nothing) that is still a panel of rows (fill).
    assert.ok(workspace.includes(`data-probe-chrome="${sheet}" data-probe-panel="${sheet}" data-probe-layer`),
      `the ${sheet} sheet lost its chrome/panel/layer markers`);
  }
});

test('the passes panel is a READING surface: asked for, one tap away, used against nothing', () => {
  assert.match(workspace, /<div id="bank-passes-panel" data-probe-chrome="passes" data-probe-panel="passes" data-probe-reading>/);
  // …and below lg it folds everything that is not a pass button: measured at
  // 360 px, the unfolded panel was ~1 500 px tall.
  assert.match(passes, /function Fold\(\{ compact, title, children \}\)/);
  for (const title of ['🧠 Semantic engine', '🧽 Watermarks', '✂ Edits', '📊 Bank overview']) {
    assert.ok(passes.includes(`<Fold compact={compact} title="${title}">`), `${title} is no longer folded below lg`);
  }
  assert.match(workspace, /compact=\{!railIsColumnNow\}/);
});

test('the passes panel does not open itself below lg', () => {
  assert.match(layout, /export const PASSES_AUTO_OPEN_MIN_PX = 1024;/);
  assert.match(layout, /export function passesPanelStartsOpen\(counts, viewportWidth = Infinity\)/);
  assert.match(workspace, /passesPanelStartsOpen\(counts, viewportWidth\(\)\)/);
});

test('the rail is a layer when it is a drawer, and a panel always', () => {
  assert.match(rail, /data-probe-chrome=\{isDrawer \? 'rail' : undefined\} data-probe-layer=\{isDrawer \? '' : undefined\}/);
  assert.match(rail, /data-probe-panel="rail"/);
});

test('the review lightbox is a layer', () => {
  assert.match(lightbox, /data-probe-chrome="review" data-probe-layer/);
});

test('the bank list names its opener, so the probe can prime the workspace', () => {
  assert.match(page, /aria-label=\{`Open the bank \$\{b\.name\}`\}/);
  assert.match(probe, /'#\/bank': \{/);
  assert.match(probe, /prime: \['\[aria-label\^="Open the bank"\]'\]/);
});

test('the header gives the fold back on a phone', () => {
  // the source-path row is a desktop gesture; the counters scroll on one line;
  // the action row scrolls on one line; a phone held sideways gets a one-row header
  assert.match(workspace, /className="hidden min-w-0 grow items-center gap-2 sm:flex \[@media\(max-height:500px\)\]:!hidden"/);
  assert.match(workspace, /flex flex-nowrap items-baseline gap-x-4 gap-y-1 overflow-x-auto border-t border-border pt-2 text-sm sm:flex-wrap sm:overflow-visible/);
  assert.match(workspace, /flex min-w-0 flex-nowrap items-center gap-2 overflow-x-auto border-t border-border pt-2 sm:flex-wrap sm:overflow-visible/);
  assert.match(workspace, /\[@media\(max-height:500px\)\]:flex \[@media\(max-height:500px\)\]:flex-nowrap/);
});

test('the probe knows what a layer is, and charges it to no budget', () => {
  // the budget exemption names both kinds of not-in-your-way surface…
  assert.match(probe, /el\.hasAttribute\('data-probe-reading'\) \? 'reading'/);
  assert.match(probe, /: el\.hasAttribute\('data-probe-layer'\) \? 'layer' : null/);
  // …and the overlap check pairs a layer with nothing
  assert.match(probe, /chrome\[i\]\.el\.hasAttribute\('data-probe-layer'\)/);
  // only VISIBLE chrome counts: a marker inside a closed <details> keeps a box
  assert.match(probe, /checkVisibility/);
});
