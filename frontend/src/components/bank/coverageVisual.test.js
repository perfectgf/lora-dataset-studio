// Reads the image Bank TREE, not one file: the Encre redesign split the
// workspace into a top bar, a filter rail, a passes panel and the grid, and a
// wiring assertion must survive a move (see bankTreeSource.js).
import { bankTreeSource } from './bankTreeSource.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { spreadReadout, spreadCoverageNote } from './coverageVisual.js';

const ws = bankTreeSource();

test('an unscored pool reads as NOT MEASURED, never as varied', () => {
  // The failure this exists to prevent: silence painted green. A bank nobody
  // scored has not been found varied — it has not been looked at.
  const r = spreadReadout({ scored: 0, similarity: null, band: 'unknown' });
  assert.equal(r.measured, false);
  assert.equal(r.label, 'Not measured');
  assert.notEqual(r.tone, 'ok');
  assert.match(r.detail, /Run ✨ Score/);
  assert.match(r.detail, /CLIP semantic index/);
  assert.match(spreadReadout({ semantic_indexed: 0 }, 'siglip2').detail,
    /Build the SigLIP 2 semantic index/);
});

test('a pool too small to judge says so instead of reporting a band', () => {
  const r = spreadReadout({ scored: 4, similarity: null, band: 'unknown' });
  assert.equal(r.measured, false);
  assert.match(r.detail, /too few/);
  assert.match(r.detail, /4 images with CLIP embeddings/);
});

test('a redundant pool is a warning and names the number', () => {
  const r = spreadReadout({ scored: 250, similarity: 0.91, band: 'redundant' });
  assert.equal(r.measured, true);
  assert.equal(r.tone, 'warn');
  assert.equal(r.percent, 91);
  assert.match(r.detail, /91% average similarity across 250 images with CLIP embeddings/);
  assert.match(r.detail, /teaches one look/);
});

test('a varied pool reads positive without pretending it is a verdict', () => {
  const r = spreadReadout({ scored: 800, similarity: 0.41, band: 'varied' });
  assert.equal(r.tone, 'ok');
  assert.match(r.detail, /41% average similarity/);
});

test('an unknown band never crashes and never warns', () => {
  const r = spreadReadout({ scored: 30, similarity: 0.5, band: 'something-new' });
  assert.ok(r.measured);
  assert.notEqual(r.tone, 'warn');
  assert.notEqual(r.tone, 'ok');
  assert.match(r.label, /not calibrated/);
  assert.equal(spreadReadout(null), null);
});

test('SigLIP2 similarity never borrows a CLIP verdict when it is uncalibrated', () => {
  const r = spreadReadout({
    engine: 'siglip2', semantic_indexed: 50, similarity: 0.88,
    band: 'redundant', calibrated: false,
  });
  assert.equal(r.measured, true);
  assert.equal(r.tone, 'info');
  assert.match(r.label, /not calibrated/);
  assert.match(r.detail, /no honest “varied\/alike” band/);
});

test('a partly-scored pool admits how much it actually read', () => {
  assert.match(spreadCoverageNote({ scored: 120 }, 500), /120 of 500/);
  assert.match(spreadCoverageNote({ semantic_indexed: 120 }, 500, 'siglip2'),
    /not indexed by SigLIP 2 yet/);
  // Fully scored: no note, because there is nothing to disclaim.
  assert.equal(spreadCoverageNote({ scored: 500 }, 500), '');
  assert.equal(spreadCoverageNote({ scored: 0 }, 500), '');
});

// --- contract: the panel keeps its honesty and its credit ------------------
// Source-text assertions: node --test cannot parse JSX.

test('the bank panel keeps the @antonp credit', () => {
  // The bank coverage card was a community idea; extending it does not make it
  // ours. This credit is TRUE and must survive any rewrite of the panel.
  assert.match(ws, /idea by @antonp/);
});

test('the panel states its three sources and the caption caveat', () => {
  assert.match(ws, /words, not pixels/);
  assert.match(ws, /still counts as a smile/);
  assert.match(ws, /Judged as a character source/);
});

test('the variety axes reuse the dataset helpers instead of a copy', () => {
  assert.match(ws, /from '\.\.\/dataset\/datasetCoverage\.js'/);
  assert.match(ws, /axisRows\(axis\)/);
});

test('the coverage refetch listens to the caption and score passes', () => {
  const effect = ws.slice(ws.indexOf('if (coverageOpen) loadCoverage()'));
  assert.match(effect.slice(0, 400), /counts\?\.captioned/);
  assert.match(effect.slice(0, 400), /counts\?\.scored/);
  assert.match(effect.slice(0, 500), /counts\?\.semantic_indexed/);
});
