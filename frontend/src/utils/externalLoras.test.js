import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizeExternalLoras, clampStrength, externalLoraPayload,
  MAX_EXTERNAL_LORAS } from './externalLoras.js';

test('normalize dedupes, drops empties, clamps and caps', () => {
  const raw = [
    { filename: 'a.safetensors', strength: 0.6, x: '3', y: -2 },
    { filename: 'a.safetensors', strength: 2 },        // dupe
    { filename: '', strength: 1 },                     // empty
    { filename: 'b.safetensors', strength: 99 },       // clamp → 2
    { filename: 'c.safetensors', strength: 'nope' },   // → 1
  ];
  assert.deepEqual(normalizeExternalLoras(raw), [
    { filename: 'a.safetensors', strength: 0.6, x: 3, y: -2 },
    { filename: 'b.safetensors', strength: 2, x: 0, y: 0 },
    { filename: 'c.safetensors', strength: 1, x: 0, y: 0 },
  ]);
  assert.equal(normalizeExternalLoras(null).length, 0);
  const many = Array.from({ length: 40 }, (_, i) => ({ filename: `l${i}.safetensors` }));
  assert.equal(normalizeExternalLoras(many).length, MAX_EXTERNAL_LORAS);
});

test('clampStrength', () => {
  assert.equal(clampStrength(-1), 0);
  assert.equal(clampStrength(2.555), 2);
  assert.equal(clampStrength(0.333), 0.33);
  assert.equal(clampStrength(undefined), 1);
});

test('payload keeps only checked nodes, engine shape', () => {
  const nodes = [
    { filename: 'a.safetensors', strength: 0.6, x: 0, y: 0 },
    { filename: 'b.safetensors', strength: 1, x: 0, y: 0 },
  ];
  assert.deepEqual(externalLoraPayload(nodes, new Set(['b.safetensors'])),
    [{ filename: 'b.safetensors', strength: 1 }]);
  assert.deepEqual(externalLoraPayload(nodes, new Set()), []);
});
