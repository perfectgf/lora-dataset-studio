import assert from 'node:assert/strict';
import test from 'node:test';

import { generationMetaRows } from './generationMetaFacts.js';

test('a stamped row renders its facts in product wording', () => {
  const rows = generationMetaRows({
    engine: 'klein', base_model: 'klein\\flux-2-klein-9b-fp8.safetensors',
    steps: 4, reference_strength: 1, aspect: '2:3',
    loras: [{ file: 'klein\\realistic.safetensors', strength: 0.8 }],
  });
  const by = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  assert.equal(by.engine, 'FLUX.2 Klein');
  assert.equal(by.base_model, 'flux-2-klein-9b-fp8');
  assert.equal(by.steps, '4');
  assert.equal(by.loras, 'realistic @ 0.8');
  assert.equal(by.aspect, '2:3');
});

test('what was not stamped produces NO row — never a guess', () => {
  const rows = generationMetaRows({ engine: 'krea' });
  assert.deepEqual(rows.map((r) => r.key), ['engine']);
  assert.deepEqual(generationMetaRows(null), []);
  assert.deepEqual(generationMetaRows('camera'), []);
  assert.deepEqual(generationMetaRows(['x']), []);
});

test('both LoRA spellings converge, and a nameless entry is dropped', () => {
  const rows = generationMetaRows({
    loras: [{ filename: 'qwen\\angles.safetensors', strength: 1 }, { strength: 2 }],
  });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].value, 'angles @ 1');
});

test('a lane that stamps MORE than the known keys is not silenced', () => {
  const rows = generationMetaRows({ engine: 'camera', seed: 7, grounding_px: 384 });
  const keys = rows.map((r) => r.key);
  assert.ok(keys.includes('grounding_px'), 'unknown scalar keys must still render');
  assert.equal(rows.find((r) => r.key === 'seed').value, '7');
});
