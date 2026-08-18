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

/* ── The picture must FIT its stage, and the canvas must sit ON the picture ──
   Both halves come from one mistake. `max-h-full` on the img resolved against
   the wrapper's own (indefinite) content height, so it computed to `none`: the
   image grew to the full stage width, overflowed downwards and covered the
   prompt/Cancel/Repair row. And because the canvas is `absolute inset-0` on the
   wrapper, it kept the WRAPPER's smaller box — measured 742 px against the
   image's 939 px — so a click 39.5 % down the visible picture painted at 50 %
   in the mask, and the bottom of the image could not be reached at all.

   `node --test` does no layout, so these read the source. The measured proof
   (img rect === canvas rect, nothing past the stage) lives in the headless
   verify run; what is pinned here is the construct that makes it true. */

const BOUNDED_H = 'max-h-[min(70vh,calc(100cqh_-_1.5rem))]';
const BOUNDED_W = 'max-w-[min(92vw,100cqw)]';
/** Only what is actually APPLIED — the prose above these elements names the old
 *  classes on purpose, and a test that reads the comments would forbid saying
 *  what went wrong. */
const classNames = (src) => (src.match(/className="[^"]*"/g) || []).join('\n');

test('the brush image is bounded by the stage, not by an unresolvable max-h-full', () => {
  // `max-h-full` needs a definite height on the parent; the wrapper's height IS
  // the image's, so it never had one. Viewport/container lengths always resolve.
  const applied = classNames(brush);
  assert.ok(!/max-h-full/.test(applied),
    'max-h-full computes to none here — it bounded nothing');
  for (const cls of [BOUNDED_H, BOUNDED_W]) {
    assert.ok((applied.match(new RegExp(cls.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length >= 2,
      `both the wrapper and the img must carry ${cls}`);
  }
  // Exactly what the box editor next door has always used — the two editors
  // share a stage and must not size themselves by different rules.
  const boxEditor = read('../dataset/WatermarkRegionEditor.jsx');
  assert.ok(boxEditor.includes(BOUNDED_H) && boxEditor.includes(BOUNDED_W));
});

test('the canvas box is the image box', () => {
  // `inset-0` is measured off the WRAPPER. That is only the displayed image
  // when the wrapper shrink-wraps it: inline-block, a single `block` img, no
  // padding, no object-fit box of its own.
  assert.ok(brush.includes(`className="relative inline-block ${BOUNDED_H} ${BOUNDED_W} leading-none"`),
    'the wrapper must shrink-wrap the image and carry the same caps');
  assert.ok(brush.includes(`className="block ${BOUNDED_H} ${BOUNDED_W} select-none"`),
    'the img must be a block with the same caps — no extra sizing of its own');
  assert.ok(!/object-(contain|cover|fill|none|scale-down)/.test(classNames(brush)),
    'an object-fit box that differs from the element box is invisible to the canvas');
  assert.match(brush, /className="absolute inset-0 h-full w-full touch-none"/);
});

test('the repair stage is a size container, so those caps mean the stage', () => {
  // Without it, 100cqh falls back to the viewport and the picture can still be
  // taller than the room between the toolbars. The other two hosts of these
  // editors already declare it.
  assert.match(dialog, /flex min-h-0 flex-1 items-center justify-center \[container-type:size\]/);
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
