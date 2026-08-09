/**
 * 🧬 The blend weight range — and the ONE thing that can silently break it.
 *
 * The head weight of a blend combination is validated by the SERVER's sweep
 * validator (lora_test_studio.build_matrix), so a browser ceiling above the
 * server's does not produce a weaker image: it produces a REFUSED run. These
 * tests read the Python file and compare the numbers, because the failure mode
 * of "someone raised one of the two" is a launch that dies with a range error
 * nobody connects to a slider.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  COMBINE_MAX_WEIGHT, COMBINE_MIN_WEIGHT, clampBlendWeight, stackWeight, stackWeightSet,
} from './loraStack.js';
import { STRENGTH_CHOICES_EXTENDED } from './constants.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const SERVICE = join(HERE, '..', '..', '..', '..', '..',
  'backend', 'app', 'services', 'lora_test_studio.py');

const pyConst = (name) => {
  const src = readFileSync(SERVICE, 'utf8');
  const m = new RegExp(`^${name} = (-?[0-9.]+)$`, 'm').exec(src);
  assert.ok(m, `${name} not found in lora_test_studio.py`);
  return Number(m[1]);
};

test('the blend ceiling is 5, not the old 2', () => {
  assert.equal(COMBINE_MAX_WEIGHT, 5);
  assert.equal(COMBINE_MIN_WEIGHT, 0);
});

test('the browser ceiling is exactly the server ceiling', () => {
  assert.equal(COMBINE_MAX_WEIGHT, pyConst('MAX_LORA_STRENGTH'));
});

test('the highest strength chip is reachable by the server', () => {
  const top = Math.max(...STRENGTH_CHOICES_EXTENDED);
  assert.equal(top, pyConst('MAX_LORA_STRENGTH'));
});

test('a weight between 2 and 5 survives instead of being clamped back to 2', () => {
  assert.equal(stackWeight({ '1:a.safetensors': 3.4 },
    { dataset_id: 1, checkpoint: 'a.safetensors' }), 3.4);
  assert.deepEqual(stackWeightSet({ '1:a.safetensors': [1, 2.5, 5] },
    { dataset_id: 1, checkpoint: 'a.safetensors' }), [1, 2.5, 5]);
});

test('anything above the ceiling is still clamped, not passed through', () => {
  assert.equal(stackWeight({ '1:a.safetensors': 40 },
    { dataset_id: 1, checkpoint: 'a.safetensors' }), 5);
});

test('typed weights are clamped and rounded to the hundredth', () => {
  assert.equal(clampBlendWeight('3.456'), 3.46);
  assert.equal(clampBlendWeight('99'), 5);
  assert.equal(clampBlendWeight('-4'), 0);
});

test('a half-typed field leaves the weight alone rather than inventing one', () => {
  // Returning 0 here would make the slider jump to the floor on every keystroke
  // that momentarily empties the box.
  assert.equal(clampBlendWeight(''), null);
  assert.equal(clampBlendWeight(null), null);
  assert.equal(clampBlendWeight('abc'), null);
});
