import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildGeneratePayload, clipSeconds, clipSummary, isRunning, SPARSE_CHOICES,
} from './videoStudioApi.js';

test('an option left off is absent from the payload, never false', () => {
  const body = buildGeneratePayload({ mode: 'i2v', prompt: ' she turns ', image: 'a.png' });
  assert.equal(body.prompt, 'she turns');
  assert.equal(body.image, 'a.png');
  for (const key of ['turbo', 'eros', 'sparse', 'latent_upscale', 'lora']) {
    assert.ok(!(key in body), `${key} should not be sent when it is off`);
  }
});

test('t2v drops the start image and keeps the aspect instead', () => {
  const body = buildGeneratePayload({
    mode: 't2v', prompt: 'a street at night', image: 'left-over.png',
    ratio: 1.77, aspect: 'portrait',
  });
  assert.ok(!('image' in body), 't2v must not carry a start frame');
  assert.ok(!('ratio' in body));
  assert.equal(body.aspect, 'portrait');
});

test('a LoRA carries its strength and its provenance', () => {
  const body = buildGeneratePayload({
    mode: 'i2v', prompt: 'p', image: 'a.png',
    lora: 'h3/lds/jessy.safetensors', loraStrength: 1.3, runId: 174, datasetId: 8,
  });
  assert.equal(body.lora, 'h3/lds/jessy.safetensors');
  assert.equal(body.lora_strength, 1.3);
  assert.equal(body.run_id, 174);
  assert.equal(body.dataset_id, 8);
});

test('seed 0 is sent — it is a seed, not an empty field', () => {
  const body = buildGeneratePayload({ mode: 't2v', prompt: 'p', seed: 0 });
  assert.equal(body.seed, 0);
  assert.ok(!('seed' in buildGeneratePayload({ mode: 't2v', prompt: 'p', seed: '' })));
});

test('every sparse choice is a level the server accepts', () => {
  // The server normalises anything it does not know to OFF, silently — which is
  // the right server behaviour and the wrong thing to discover from a render.
  const accepted = new Set(['', 'default', 'conservative', 'max']);
  for (const c of SPARSE_CHOICES) {
    assert.ok(accepted.has(c.value), `unknown sparse level "${c.value}"`);
    assert.ok(c.label && c.hint, `sparse level "${c.value}" needs a label and a hint`);
  }
});

test('clip length counts intervals, not frames', () => {
  assert.equal(clipSeconds(121, 24), 5);       // the lane's own cross-check
  assert.equal(clipSeconds(0, 24), null);
  assert.equal(clipSeconds(56, 0), null);
});

test('the summary names what differed and stays quiet about what did not', () => {
  const line = clipSummary({
    lora: 'h3\\lds\\jessy_2000.safetensors', lora_strength: 1.3, turbo: true,
    sparse: 'conservative', steps: 6, seed: 42, latent_upscale: false, eros: false,
  });
  assert.match(line, /jessy_2000 @ 1\.3/);
  assert.match(line, /⚡ turbo/);
  assert.match(line, /sparse conservative/);
  assert.ok(!line.includes('upscale'), 'an option that was off must not be listed');
  assert.ok(!line.includes('10Eros'));
  assert.match(clipSummary({ steps: 20, seed: 1 }), /no LoRA/);
});

test('running is one predicate', () => {
  assert.equal(isRunning({ status: 'pending' }), true);
  assert.equal(isRunning({ status: 'done' }), false);
  assert.equal(isRunning(null), false);
});
