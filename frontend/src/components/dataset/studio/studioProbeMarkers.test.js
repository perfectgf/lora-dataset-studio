/* 📐 The markers the responsive probe measures the Test Studio by.
 *
 * Same contract as bankProbeMarkers.test.js: `scripts/responsiveProbe.mjs`
 * finds its surfaces by attribute, and these assertions keep the attributes
 * in place — the probe is the only thing that can SEE the layout, this file
 * only guarantees it is still looking at the right elements.
 *
 * First measured 2026-08-23: 48 violations on the first run (the bottom bar's
 * 27-px shortcut chips and its Generate button, at every width below lg), 0
 * after. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8');
const shell = read('./StudioShell.jsx');
const actionBar = read('./StudioActionBar.jsx');
const setup = read('./RunSetupPanel.jsx');
const picker = read('./LoraPicker.jsx');
const probe = read('../../../../scripts/responsiveProbe.mjs');

test('the Studio chrome and panels are marked for the responsive probe', () => {
  assert.match(shell, /<header data-probe-chrome="header"/);
  assert.match(actionBar, /<nav aria-label="Studio quick navigation" data-probe-chrome="action-bar"/);
  assert.match(setup, /data-probe-panel="setup"/);
  assert.match(picker, /data-probe-panel="picker"/);
});

test('the bottom bar is finger-sized below lg, untouched on a desktop', () => {
  // the shortcut chips and the Generate button both carry the pattern
  assert.equal((actionBar.match(/min-h-10 lg:min-h-0/g) || []).length, 2);
});

test('the probe reaches the Studio through its id-carrying route', () => {
  assert.match(probe, /'#\/dataset\/studio': \{/);
  // `#/dataset/studio/1` must land on the Studio spec, not on `#/datasets` and
  // not on nothing: the lookup is longest-prefix over segment boundaries.
  assert.match(probe, /const hashPath = \(\(args\.url\.split\('#'\)\[1\] \|\| '\/'\)\.split\('\?'\)\[0\]\);/);
  assert.match(probe, /return hashPath === p \|\| hashPath\.startsWith\(p \+ '\/'\);/);
  assert.match(probe, /\.sort\(\(a, b\) => b\.length - a\.length\)\[0\] \|\| null;/);
  // an unknown page is still measured at rest — never skipped silently
  assert.match(probe, /const pageSpec = \(route && PAGES\[route\]\) \|\| UNKNOWN_PAGE;/);
});

test('the shortcut state drives the bar the users drive', () => {
  assert.match(probe, /\{ name: 'shortcut', open: \['\[data-probe-chrome="action-bar"\] button'\] \}/);
});
