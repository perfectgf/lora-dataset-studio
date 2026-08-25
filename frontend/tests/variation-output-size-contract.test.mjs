/* The ONE output-size dial of the Generate-variations panel.

   It is shared: Klein and Krea render a dataset's shots at the same budget, so
   the control cannot live inside either engine's own <details> block. These
   assertions are about the two things a user can actually be misled by — the
   number the panel promises and where the panel puts the control. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  VARIATION_MP_MIN, VARIATION_MP_MAX, VARIATION_MP_STEP,
  clampVariationMegapixels, variationCanvas, variationSizeDescription,
  variationOutputSizePayload,
} from '../src/utils/variationOutputSize.js';

const panel = readFileSync(
  new URL('../src/components/dataset/VariationCatalog.jsx', import.meta.url), 'utf8');

test('the dial spans the researched range, ceiling included', () => {
  assert.equal(VARIATION_MP_MIN, 0.5);
  // 2.0 is where the edit models start to drift (backend MAX_OUTPUT_MP) AND
  // Klein's historical hardcoded value: the default must be reachable.
  assert.equal(VARIATION_MP_MAX, 2.0);
  assert.equal(VARIATION_MP_STEP, 0.1);
});

test('a value from a stale backend or a broken config still lands in range', () => {
  assert.equal(clampVariationMegapixels(1.2), 1.2);
  assert.equal(clampVariationMegapixels(9), 2.0);
  assert.equal(clampVariationMegapixels(0.01), 0.5);
  assert.equal(clampVariationMegapixels('nonsense'), 2.0);
  assert.equal(clampVariationMegapixels(undefined, 1.5), 1.5);
  assert.equal(clampVariationMegapixels(null, 'junk'), 2.0);
  // The slider steps by 0.1; a config holding 1.234567 must not print as that.
  assert.equal(clampVariationMegapixels(1.234567), 1.2);
});

/* Pinned against backend/app/services/output_geometry.fit_output_size — the
   panel states a pixel size, so it has to be the size that is actually
   rendered. backend/tests/test_variation_output_size.py asserts the same pairs
   from the real function, so a change to either side turns one of them red. */
test('the canvas the panel announces is the canvas the backend renders', () => {
  assert.deepEqual(variationCanvas(2.0, '3:4'), [1216, 1632]);
  assert.deepEqual(variationCanvas(2.0, '1:1'), [1408, 1408]);
  assert.deepEqual(variationCanvas(1.0, '3:4'), [864, 1152]);
  assert.deepEqual(variationCanvas(0.5, '16:9'), [944, 528]);
});

test('the description names a size, not just a number', () => {
  const text = variationSizeDescription(2.0);
  assert.match(text, /2\.0 MP/);
  assert.match(text, /1216\s*×\s*1632/, 'the portrait card size must be spelled out');
  // The shipped value has to be recognisable as such, or nobody dares move it.
  assert.match(variationSizeDescription(2.0), /default/i);
  assert.match(variationSizeDescription(0.5), /fast|quick/i);
});

test('a drag writes only its own key', () => {
  assert.deepEqual(variationOutputSizePayload({ output_megapixels: 1.5 }),
    { config: { variations: { output_megapixels: 1.5 } } });
});

test('the dial sits outside both engine tuning blocks', () => {
  const marker = 'variation-output-size-dial';
  assert.ok(panel.includes(marker), 'the dial must be rendered by the panel');
  const dial = panel.indexOf(marker);
  const kleinBlock = panel.indexOf('/>Klein tuning');
  const presets = panel.indexOf("uppercase\">Presets</span>");
  assert.ok(dial < kleinBlock,
    'a shared setting cannot live inside the Klein-only <details>');
  assert.ok(dial < presets, 'the dial belongs above the shot cards');
});

test('the panel shows the dial for either local engine, never only for one', () => {
  const guard = panel.indexOf('{(isKlein || isKrea)');
  assert.ok(guard > -1, 'the shared dial needs a guard naming BOTH local engines');
  const dial = panel.indexOf('variation-output-size-dial');
  assert.ok(guard < dial && dial < panel.indexOf('/>Klein tuning'),
    'the dial must sit inside that shared guard, above the engine tuning blocks');
});
