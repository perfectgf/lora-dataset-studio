/* ⚠ One engine, one answer to "why not".
 *
 * `utils/localEngineReason.js` exists because a second inline copy of a reason is
 * how "⚠ Configure ComfyUI in Settings" ends up worded one way in one dialog and
 * another way two clicks later, for the same missing file. The generation picker
 * and the ✦ Edit modal read it. FOUR watermark surfaces and the concept scraper
 * did not: they each carried a hand-written catch-all —
 *
 *     "Klein inpainting needs ComfyUI running + the Klein models (Setup ▸ ComfyUI)"
 *     "Klein is not ready in this setup."
 *
 * — which names no cause at all. It is the same defect the Setup screen had (a ✓
 * posted without checking), pointed the other way: a refusal stated without a
 * reason. Worse, it is confidently WRONG in the case that actually happens — a
 * present-but-corrupted 9.5 GB weight sends the user off to re-check ComfyUI and
 * re-download files that are already there.
 *
 * Read as text: `node --test` cannot parse JSX, and nothing throws when a screen
 * quietly grows its own sentence again.
 */
import assert from 'node:assert/strict';
import { readSource } from './support/readSource.mjs'
import test from 'node:test';

const read = (rel) => readSource(`src/${rel}`)

const SURFACES = [
  ['bank/BankWatermarkPanel.jsx', 'the bank cleaner\'s Level-3 engine toggle'],
  ['dataset/DatasetWorkspace.jsx', 'the dataset watermark engine toggle'],
  ['dataset/WatermarkReviewLightbox.jsx', 'the per-image review engine toggle'],
  ['dataset/ConceptSourcesPanel.jsx', 'the small-image Klein rescue checkbox'],
];

for (const [file, what] of SURFACES) {
  test(`${what} explains a Klein refusal from the shared reason`, () => {
    const src = read(`components/${file}`);
    assert.match(src, /localEngineUnavailableReason/,
      `${file} must ask utils/localEngineReason, not invent its own sentence`);
    assert.match(src, /localEngineUnavailableReason\('klein', caps\)/);
  });
}

test('the bank cleaner passes the reason down to its JSX-free state helper', () => {
  // bankWatermark.js decides which Level-3 button is live and why not; the panel
  // is only the shell, so the reason has to travel with the verdict.
  const panel = read('components/bank/BankWatermarkPanel.jsx');
  assert.match(panel, /kleinReason,/);
  const helper = read('components/bank/bankWatermark.js');
  assert.match(helper, /kleinReason = null/);
  assert.match(helper, /kleinReason\s*\n?\s*\|\|/);
});

test('the concept rescue checkbox no longer states a verdict with no cause', () => {
  const src = read('components/dataset/ConceptSourcesPanel.jsx');
  assert.ok(!src.includes('Klein is not ready in this setup.'),
    'that sentence named the verdict and hid the one thing the user needed');
  assert.match(src, /kleinReason \?/);
});

/* The catch-alls may survive as a LAST-RESORT `||` fallback (a caller with no
   capabilities payload has nothing better to say), but never as the only text on
   the branch — that is the state this test was written to end. */
test('no Klein catch-all is stated outright — each one is a `||` fallback', () => {
  const files = [...SURFACES.map(([f]) => `components/${f}`),
                 'components/bank/bankWatermark.js'];
  for (const file of files) {
    const src = read(file);
    const needle = /Klein inpaint(?:ing)? needs ComfyUI/g;
    for (let m = needle.exec(src); m; m = needle.exec(src)) {
      // The 120 characters before it must carry the `||` that makes it a
      // last resort rather than the answer.
      const before = src.slice(Math.max(0, m.index - 120), m.index);
      assert.ok(before.includes('||'),
        `${file}: the catch-all at ${m.index} is the only text on its branch`);
    }
  }
});
