/* 🪪 The dataset's reference face, on the ◉ LoRA Canvas.

   The board's whole job is judging whether a checkpoint got the likeness right,
   and it showed every render and never the person — so the comparison happened
   from memory. The reference now opens each lane's header.

   Pinned as a SOURCE contract because `node --test` cannot parse JSX: the parts
   that would silently disappear in a rewrite are the gate (a concept or a style
   dataset has no reference face and must not be shown an empty frame), the URL
   shape, and the fact that the lane is actually handed the two fields. None of
   those throws when it breaks. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const canvas = fs.readFileSync(new URL('./LineageCanvas.jsx', import.meta.url), 'utf8');
const page = fs.readFileSync(
  new URL('../../pages/CanvasPage.jsx', import.meta.url), 'utf8');
const lightbox = fs.readFileSync(
  new URL('../shared/GeneratedImageLightbox.jsx', import.meta.url), 'utf8');

test('the lane header shows the reference only for a CHARACTER dataset', () => {
  assert.match(canvas, /lane\.kind !== 'concept' && lane\.kind !== 'style'/);
  assert.match(canvas, /Boolean\(lane\.refFilename\)/);
});

test('the reference URL is the dataset image route, with the name encoded', () => {
  // A reference filename is generated, but the route is the shared one and a
  // raw filename in a URL is how a path gets broken by the first odd character.
  assert.match(canvas,
    /\/api\/dataset\/\$\{lane\.datasetId\}\/img\/\$\{encodeURIComponent\(lane\.refFilename\)\}/);
});

test('the reference is enlargeable, and labelled for a screen reader', () => {
  assert.match(canvas, /onZoomRef\?\.\(\{ url: refUrl, name: lane\.name \}\)/);
  assert.match(canvas, /<LaneHeader lane=\{lane\} onZoomRef=\{setRefZoom\} \/>/);
  assert.match(canvas, /aria-label=\{`Reference image of \$\{lane\.name\}/);
  // Its OWN lightbox: a reference is not a generated image and has no seed,
  // prompt or settings to put under it.
  assert.match(canvas, /alt=\{`Reference image of \$\{refZoom\?\.name/);
  // …and NO generation facts: a reference has no seed, sampler or prompt, and
  // the column announced "SEED —" plus a Download named after a run and a step
  // this picture does not have.
  assert.match(canvas, /facts=\{false\} onClose=\{\(\) => setRefZoom\(null\)\}/);
  // `facts` defaults to true and the component takes an onClose. Matched loosely
  // on purpose: this pins the DEFAULT, not the arity — the signature has since
  // grown the ✨ improve opt-in, and a test that breaks every time a prop is
  // added teaches nothing about the reference image it is here to protect.
  assert.match(lightbox, /facts = true, onClose[,}]/);
  // The column is drawn only when there ARE facts. It now also folds on demand
  // (⤢ gives the picture the whole screen), so the condition is one derived
  // boolean rather than `facts` alone — what is pinned here is that `facts`
  // still gates it, which is the reference image's half of the contract: a 🪪
  // face has no seed, no sampler and no prompt, and an empty table is a wrong
  // answer rather than a neutral default.
  assert.match(lightbox, /const showFacts = facts && factsOpen;/);
  assert.match(lightbox, /\{showFacts && \(\s*<aside/);
});

test('the reference button opts out of the board gesture', () => {
  // Without this the frame captures the pointerdown and the click that follows
  // is retargeted away from the button — it renders, it is correct, and
  // clicking it does nothing. Verified in a browser; pinned here because it is
  // invisible to every other test.
  assert.match(canvas, /<button type="button" data-canvas-control/);
});

test('the lane carries the reference and the kind from the canvas index', () => {
  assert.match(page, /refFilename: row\?\.ref_filename \|\| null/);
  assert.match(page, /kind: row\?\.kind \|\| 'character'/);
});
