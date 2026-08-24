/* 📐 The Gallery's responsive-probe markers — same contract as the canvas,
 * bank, dataset and studio siblings: the probe measures by ATTRIBUTE, so the
 * one silent failure mode is an attribute quietly deleted, after which the
 * probe measures nothing and reports clean. These assertions run on every
 * commit; the probe itself only runs when somebody points it at a screen.
 *
 * ⚠️ This file cannot tell you the layout is good — only that the thing which
 * CAN (scripts/responsiveProbe.mjs on #/gallery) is still pointed at the
 * right elements. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const page = fs.readFileSync(new URL('./GalleryPage.jsx', import.meta.url), 'utf8');
const lightbox = fs.readFileSync(
  new URL('../components/shared/GeneratedImageLightbox.jsx', import.meta.url), 'utf8');
const probe = fs.readFileSync(
  new URL('../../scripts/responsiveProbe.mjs', import.meta.url), 'utf8');

test('both fixed surfaces are marked for the budget and overlap checks', () => {
  // The filter rail is also what proves the page painted at all — the probe's
  // chrome wait keys on it, so losing the marker turns every gallery run into
  // the thin painted-root fallback.
  assert.match(page, /data-probe-chrome="gallery-filters"/);
  assert.match(page, /data-probe-chrome="gallery-bar"/);
});

test('the overlays that cover the page BY DESIGN are layers, not chrome', () => {
  assert.match(page, /data-testid="gallery-confirm" data-probe-layer/);
  assert.match(lightbox, /data-testid="generated-image-lightbox" data-probe-layer/);
});

test('the probe knows the page: its states and the selector that means "data arrived"', () => {
  assert.match(probe, /'#\/gallery': \{/);
  assert.match(probe, /ready: '\[data-testid="gallery-zoom"\]'/);
  for (const opener of ['[data-testid="gallery-zoom"]',
    '[data-testid="gallery-select-toggle"]']) {
    assert.ok(probe.includes(`open: ['${opener}']`),
      `the probe lost its gallery state opener ${opener}`);
  }
  // …and the openers still exist in the page, or the states silently skip.
  assert.match(page, /data-testid=\{picking \? 'gallery-pick' : 'gallery-zoom'\}/);
  assert.match(page, /data-testid="gallery-select-toggle"/);
});
