import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildGeneratePayload, clipSeconds, clipSummary, isRunning, SPARSE_CHOICES, studioFrameChoices,
}  from './videoStudioApi.js';

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

// --- the sampling steps, once they became reachable (2026-09-01) --------------

test('an explicit step count travels; auto sends nothing at all', () => {
  // The server has always accepted `steps` and always let it win over turbo's
  // own six — the panel simply never offered the dial, so the one number that
  // trades time for fidelity was the one nobody could turn.
  const withSteps = buildGeneratePayload({
    mode: 't2v', prompt: 'a street', turbo: true, steps: 12,
  })
  assert.equal(withSteps.steps, 12)
  assert.equal(withSteps.turbo, true)
  // Auto is the ABSENCE of the key: the server then applies the count for the
  // mode in force, and nothing here claims a choice nobody made.
  const auto = buildGeneratePayload({ mode: 't2v', prompt: 'a street', turbo: true, steps: '' })
  assert.equal('steps' in auto, false)
})

// --- the studio's own clip lengths, not training's (2026-09-01) ---------------

test('the length list reaches the model, not the training catalogue', () => {
  // The dropdown was built from `frame_choices` — the TRAINING ladder, which
  // stops at 209 frames (8.67s) because that is where training lengths stop
  // being useful. The server has always accepted up to 362 (15.04s) and says
  // so in its own comment; the list on screen was the wrong table.
  const l = studioFrameChoices({ frames_min: 22, frames_max: 362 })
  assert.equal(l[0], 22)
  assert.equal(l[l.length - 1], 362)
  // 362 frames at 24 fps is 15.04s — the model's own reach.
  assert.equal(clipSeconds(362, 24), 15.04)
  // Every rung is legal for H3's VAE: 17 pixel frames per chunk, so ≡ 5 mod 17.
  assert.ok(l.every((f) => f % 17 === 5))
  // No duplicates, ascending.
  assert.deepEqual([...l].sort((a, b) => a - b), l)
  assert.equal(new Set(l).size, l.length)
})

test('the length list falls back rather than inventing lengths', () => {
  const fallback = studioFrameChoices({ frame_choices: [39, 56] })
  assert.ok(fallback.length > 0)
  assert.ok(fallback.every((f) => f % 17 === 5 || [39, 56].includes(f)))
})
