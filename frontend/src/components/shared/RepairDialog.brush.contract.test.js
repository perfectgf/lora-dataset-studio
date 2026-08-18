/**
 * ✦ Repair now has TWO shapes — a drawn box and a painted brush — behind ONE
 * button. `node --test` cannot parse JSX, so what these are WIRED to is pinned
 * by reading the source: a brush that silently posted boxes, or a second entry
 * point growing in an action bar, would leave any behavioural test green.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const dialog = read('./RepairDialog.jsx');
const brush = read('./InpaintBrushEditor.jsx');
const review = read('../dataset/WatermarkReviewLightbox.jsx');
const generated = read('./GeneratedImageLightbox.jsx');
const canvas = read('../canvas/LineageCanvas.jsx');
const hook = read('../../hooks/useDataset.js');

test('the brush lives inside the repair dialog, not behind a new button', () => {
  assert.match(dialog, /import InpaintBrushEditor, \{ maskPngFromCanvas \}/);
  // ONE entry point per surface. A third door to the same place is exactly what
  // the phone-layout wave removed, and it must not grow back here.
  for (const [name, src] of [['watermark review', review], ['generated', generated]]) {
    assert.equal((src.match(/<RepairDialog\b/g) || []).length, 1,
      `${name}: exactly one RepairDialog`);
  }
  assert.ok(!review.includes('🖌 Touch up') && !generated.includes('🖌 Touch up'),
    'the brush must not become a separate action-bar button');
});

test('a box and a brush are mutually exclusive on the wire', () => {
  // The server picks its geometry from WHICH key arrived, so sending both would
  // make the choice ambiguous at the only place it is made.
  assert.match(dialog, /mode === 'brush'\s*\n?\s*\?\s*\{ mask, prompt: prompt\.trim\(\) \}\s*\n?\s*:\s*\{ boxes: regions, prompt: prompt\.trim\(\) \}/);
});

test('the box stays the default — the gesture that shipped is not replaced', () => {
  assert.match(dialog, /useState\('box'\)/);
});

test('an unpainted brush is refused in the dialog, before the round-trip', () => {
  // Klein would happily spend a GPU minute returning the same image.
  assert.match(dialog, /if \(mode === 'brush' && !mask\) \{/);
  assert.match(dialog, /brush \? painted : regions\.length > 0/);
});

test('the mask is read at submit time, not held in React state', () => {
  // It is megabytes of pixels; keeping it in state re-encodes it on every stroke.
  assert.match(dialog, /const mask = mode === 'brush' \? maskPngFromCanvas\(brushCanvasRef\.current\) : null;/);
  assert.ok(!/useState\(\s*maskPngFromCanvas/.test(dialog));
});

test('the mask leaves the browser hard black-and-white', () => {
  // A soft alpha edge would reach the server as "repaint this a bit", which no
  // lane means — the backend thresholds nothing.
  assert.match(brush, /const on = src\.data\[i \+ 3\] > 8;/);
  assert.match(brush, /const v = on \? 255 : 0;/);
  assert.match(brush, /if \(!painted\) return null;/);
});

test('the canvas is the image at its NATURAL size, not its displayed size', () => {
  // The mask has to line up with the file on disk, whatever the screen did.
  assert.match(brush, /canvas\.width = image\.naturalWidth;/);
  assert.match(brush, /canvas\.height = image\.naturalHeight;/);
});

test('painting works under a finger', () => {
  for (const needle of ['onPointerDown', 'onPointerMove', 'onPointerCancel',
    'setPointerCapture', 'touch-none']) {
    assert.ok(brush.includes(needle), `the brush needs ${needle}`);
  }
});

test('every surface forwards the mask all the way to its route', () => {
  assert.match(hook, /repairImageRegion = useCallback\(async \(imageId, prompt, boxes, mask = null\)/);
  assert.match(hook, /\{ prompt, boxes, mask \}/);
  assert.match(review, /submitRepair = useCallback\(async \(\{ boxes, mask, prompt \}\)/);
  assert.match(generated, /onSubmit=\{\(\{ boxes, mask, prompt \}\) => onRepair\.submit\(img\.id, boxes, prompt, mask\)\}/);
  assert.match(canvas, /postJson\(`\/api\/studio\/image\/\$\{imageId\}\/repair`, \{ boxes, prompt, mask \}\)/);
});

test('the contribution is credited where the code lives', () => {
  // Repo rule: community work names its author in the source it landed in.
  for (const [name, src] of [['brush editor', brush], ['dialog', dialog]]) {
    assert.match(src, /JacobArrow/, `${name} must credit its contributor`);
  }
});

/* ── The dialog is a LAYER, and a layer must not act on the one beneath it ──
   Reported from the watermark review: clicking the description field threw the
   user back to the dataset. Every host mounts this dialog inside its own
   overlay (so it inherits the stacking context), and those overlays close on a
   backdrop click — so an event the dialog does not stop is a close. */

test('a click inside the dialog never reaches the overlay behind it', () => {
  assert.match(dialog, /onClick=\{\(e\) => e\.stopPropagation\(\)\}/);
  // The host DOES close on a backdrop click — that is what makes this load-bearing.
  assert.match(review, /className="fixed inset-0 z-\[9997\][^"]*" onClick=\{close\}/);
});

test('Escape peels one layer, not two', () => {
  // Both the dialog and its hosts listen on `window`, so both fire unless the
  // host stands down while the dialog is up.
  assert.match(dialog, /if \(e\.key === 'Escape' && !busy\) onClose\(\)/);
  assert.match(review, /if \(repairOpen\) return;/);
  assert.match(generated, /if \(e\.key === 'Escape' && !repairOpen\) onClose\?\.\(\)/);
});

test('the guards are re-read when the dialog opens, not captured stale', () => {
  // A listener registered once with repairOpen=false would keep closing forever.
  assert.match(review, /doDismiss, doReject, repairOpen\]\);/);
  assert.match(generated, /\}, \[img, onClose, repairOpen\]\);/);
});
