/* ✨ Upscale & improve on the ◉ Canvas board — the decision, and the wiring that
   carries it to two of the shared lightbox's three hosts: the board, and the
   checkpoint / run gallery the result is delivered to. The third (a pill's
   preview) must gain nothing, which is asserted here and again in the DOM in
   tests/gallery-improve-render.test.mjs.

   The expensive bug this file guards is not visual. The board's pictures are
   `lora_test_image` rows and the dataset improve route resolves a
   `face_dataset_image`: two tables, two independent id spaces. Wiring the board
   to the dataset route does not throw, does not 404, and improves a real but
   UNRELATED picture. So the route is asserted here character by character. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  CANVAS_IMAGE_IMPROVE, canImproveCanvasImage, canvasImproveLaunchMessage,
  canvasImproveRefusal, isCanvasImproveRow,
} from './canvasImprove.js';

const lightbox = fs.readFileSync(
  new URL('../components/shared/GeneratedImageLightbox.jsx', import.meta.url), 'utf8');
const canvas = fs.readFileSync(
  new URL('../components/canvas/LineageCanvas.jsx', import.meta.url), 'utf8');
const gallery = fs.readFileSync(
  new URL('../components/shared/CheckpointGalleryPanel.jsx', import.meta.url), 'utf8');
const preview = fs.readFileSync(
  new URL('../components/dataset/PreviewLightbox.jsx', import.meta.url), 'utf8');
const hook = fs.readFileSync(
  new URL('../hooks/useCanvasImageImprove.js', import.meta.url), 'utf8');

// --- the decision -----------------------------------------------------------

test('a picture with a library row can be improved', () => {
  assert.equal(canvasImproveRefusal({ id: 12, dataset_id: 3 }), null);
  assert.equal(canImproveCanvasImage({ id: 12, dataset_id: 3 }), true);
});

test('a picture the board holds only as a URL is refused, with a reason', () => {
  // The lane's reference face and a pill's preview: no row, no id to send.
  // Without this guard the ✨ group would light up on a preview whose only
  // number is a STEP, and post it as if it were an image id.
  for (const img of [null, undefined, {}, { url: '/x.png', step: 900 }]) {
    assert.match(canvasImproveRefusal(img), /no library entry/);
    assert.equal(canImproveCanvasImage(img), false);
  }
});

test('an improvement CAN be improved again — passes chain', () => {
  // The server lifted its derivation guard (improves chain: Klein detail then
  // SeedVR2 resolution on one picture); the board must not keep a stale
  // mirror of the refusal it used to give.
  const derived = { id: 9, derivation_kind: CANVAS_IMAGE_IMPROVE };
  assert.equal(isCanvasImproveRow(derived), true);
  assert.equal(canvasImproveRefusal(derived), null);
  assert.equal(canImproveCanvasImage(derived), true);
  // An ordinary render is NOT an improve row, whatever else it carries.
  assert.equal(isCanvasImproveRow({ id: 9, derivation_kind: null }), false);
});

test('the stored derivation kind matches the backend, byte for byte', () => {
  // Written into user databases: renaming it strands every existing row, so it
  // is pinned here as a value, not as a variable.
  assert.equal(CANVAS_IMAGE_IMPROVE, 'canvas_image_improve');
});

test('the launch message says WHERE the result will appear', () => {
  // Nothing on the board changes when the pass is queued, so a bare "started"
  // reads as a dead click. The message names the gallery and promises the
  // original is untouched.
  const msg = canvasImproveLaunchMessage('SeedVR2');
  assert.match(msg, /^SeedVR2 started/);
  assert.match(msg, /gallery/);
  assert.match(msg, /untouched/);
});

// --- the shared lightbox: opt-in, never inferred -----------------------------

test('the improve group renders ONLY when the host passes onImprove', () => {
  assert.match(lightbox, /\{onImprove && \(\s*<ImproveActions/);
  // Default null → the two other hosts render exactly what they always did.
  assert.match(lightbox, /onImprove = null/);
});

test('the improve group is its own component, so useCapabilities never runs for the other hosts', () => {
  // useCapabilities() THROWS outside its provider. Calling it in the lightbox
  // body would make every host pay for a feature only one of them opted into.
  assert.match(lightbox, /function ImproveActions\(\{/);
  const improveBody = lightbox.slice(lightbox.indexOf('function ImproveActions'),
    lightbox.indexOf('export default function GeneratedImageLightbox'));
  assert.match(improveBody, /const \{ caps \} = useCapabilities\(\)/);
  // The shared component's own body must never call it — that is what keeps the
  // gallery and the pill preview working with nothing added.
  const shared = lightbox.slice(
    lightbox.indexOf('export default function GeneratedImageLightbox'));
  assert.doesNotMatch(shared, /useCapabilities\(/);
});

test('the engines, their labels and the Klein note are REUSED, not restated', () => {
  assert.match(lightbox, /lightboxImproveButtons\(\{/);
  // The Klein note moved INTO the shared ImproveModal (the ✨ button opens
  // it) — still one component, reused, never restated; the viewer now mounts
  // the modal instead of the prose.
  assert.match(lightbox, /import ImproveModal from '\.\/ImproveModal'/);
  const modal = fs.readFileSync(
    new URL('../components/shared/ImproveModal.jsx', import.meta.url), 'utf8');
  // One component, imported from where it lives — the modal now also takes
  // its two module-level handles (flush + settle) from that same file, so a
  // run cannot start on dials that have not landed yet. Still one source of
  // the panel; the assertion follows the import, not its shape.
  assert.match(modal, /import KleinImproveNote[\s\S]{0,120}from '\.\.\/dataset\/KleinImproveNote'/);
  // The pressed engine still travels to the handler for the direct engines.
  assert.match(lightbox, /onImprove\(img\.id, engineId\)/);
  // No second copy of the engine names in this file.
  assert.doesNotMatch(lightbox, /SeedVR2/);
});

test('the improve buttons sit beside Download and go full width on a phone', () => {
  const footer = lightbox.slice(lightbox.indexOf('lightbox-download'));
  assert.match(footer, /lightbox-download[^]{0,1000}<ImproveActions/);
  assert.match(lightbox, /min-h-9 w-full rounded-lg border border-indigo-400\/50/);
  assert.match(lightbox, /sm:w-auto/);
});

// --- the three hosts --------------------------------------------------------

test('the CANVAS passes the handler and the dataset the Klein note needs', () => {
  assert.match(canvas, /onImprove=\{canImproveCanvasImage\(pinnedZoom\) \? handleImproveCanvasImage : undefined\}/);
  assert.match(canvas, /datasetId=\{pinnedZoom\?\.dataset_id \?\? null\}/);
});

test('the CHECKPOINT GALLERY passes the same handler, on the same terms', () => {
  /* The gallery is where an improvement is DELIVERED — the pass copies its
     source's record_id/step, so the result is a row of this very panel. Offering
     the button on the board but not here meant the one screen where the result
     appears was the one screen it could not be started from.
     Same guard, same prop, same dataset for the Klein note: what differs between
     the two hosts is the name of the zoomed row, and nothing else. */
  assert.match(gallery, /onImprove=\{canImproveCanvasImage\(zoom\) \? improveImage : undefined\}/);
  assert.match(gallery, /datasetId=\{zoom\?\.dataset_id \?\? null\}/);
  // Both scopes of the panel — a checkpoint's gallery and a whole run's — are
  // this ONE component, so the run gallery gains it by construction.
  assert.match(gallery, /export default function CheckpointGalleryPanel/);
});

test('both hosts call ONE handler — the route exists in a single file', () => {
  /* THE assertion of this file. `/api/dataset/image/<id>/improve` resolves a
     face_dataset_image; a `lora_test_image` id sent there does not 404, it hits
     a real but UNRELATED picture and improves it silently. Two hosts meant two
     chances to write that route wrong, so there is one implementation and the
     hosts import it. */
  assert.match(hook, /postJson\(`\/api\/canvas\/image\/\$\{imageId\}\/improve`/);
  // EVERY call this hook makes, not just the one spelled out above: the wrong
  // route is one copy-paste away and it fails silently, so the whole set is
  // pinned. (Read from the call sites, so the prose above may name the trap.)
  const called = [...hook.matchAll(/(?:postJson|apiFetch)\(\s*`([^`]+)`/g)].map((m) => m[1]);
  assert.deepEqual(called, ['/api/canvas/image/${imageId}/improve']);
  assert.match(hook, /engineId \? \{ engine: engineId \} : \{\}/);
  // The engine the SERVER echoes names the toast, so a stale tab cannot claim
  // the wrong pass ran.
  assert.match(hook, /improveEngine\(d\.engine\)\.label/);
  for (const [name, host] of [['LineageCanvas', canvas], ['CheckpointGalleryPanel', gallery]]) {
    assert.match(host, /useCanvasImageImprove\(\)/, `${name} must use the shared handler`);
    assert.doesNotMatch(host, /\/improve`/, `${name} restates the improve route`);
  }
});

test('a failure is said out loud, never swallowed', () => {
  assert.match(hook, /toast\.error\(d\?\.error \|\| 'Could not start the improvement'\)/);
  assert.match(hook, /catch \(err\)[^]{0,120}toast\.error/);
});

test('the reference-face lightbox is NOT given the action', () => {
  // Same component, same file, one line apart — and a reference face has no
  // library row at all. `facts={false}` marks it; no onImprove must reach it.
  const refMount = canvas.slice(canvas.indexOf('img={refZoom'));
  const refBlock = refMount.slice(0, refMount.indexOf('/>') + 2);
  assert.doesNotMatch(refBlock, /onImprove/);
});

test('the pill preview is untouched', () => {
  // Zero regression on the third host: a pill's preview is a URL and a STEP,
  // held by a lineage node — there is no row to improve and no id to send, so
  // it passes nothing and renders exactly what it always did. (Proved in the
  // DOM as well, in tests/gallery-improve-render.test.mjs.)
  assert.doesNotMatch(preview, /onImprove/);
});
