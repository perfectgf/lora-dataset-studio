import test from 'node:test';
import assert from 'node:assert/strict';
import {
  PROMPT_FOLD_CHARS, checkpointFileLabel, extraLoraSummary, imageFactsLine,
  imageHeadlineFacts, imagePromptBlocks, imageSettingFacts, promptFold,
} from './generatedImageFacts.js';

const IMG = {
  id: 1, url: '/a.png', step: 2500, seed: 208607443, strength: 0,
  checkpoint: 'z image\\Ada-2500.safetensors', base_model: 'zturbo.safetensors',
  sampler: 'euler', scheduler: 'simple', cfg: 3.5, steps: 20, aspect: '9:16',
  extra_loras: '[{"filename":"style\\\\Film.safetensors","strength":0.4}]',
  face_score: 0.6789123, created_at: '2026-07-27T10:11:12',
  prompt: 'a portrait', negative: 'blurry',
};

test('the three facts you actually look for come first, and the seed is copyable', () => {
  const head = imageHeadlineFacts(IMG);
  assert.deepEqual(head.map((f) => f.key), ['step', 'seed', 'strength']);
  assert.equal(head[1].copy, '208607443');
});

test('a strength of 0 is a fact, not an absence', () => {
  // `0` is falsy, and it is a value the run panel really shows — a reported
  // screenshot had `strength 0`, which the old code dropped from the line.
  assert.equal(imageHeadlineFacts({ seed: 1, strength: 0 })
    .some((f) => f.key === 'strength' && f.value === '0'), true);
  assert.equal(imageHeadlineFacts({ seed: 1, step: 0 })
    .some((f) => f.key === 'step' && f.value === '0'), true);
});

test('a seed is never grouped or localised — it has to be pasteable', () => {
  const seed = imageHeadlineFacts({ seed: 208607443 })[0];
  assert.equal(seed.value, '208607443');
  assert.equal(/[^0-9]/.test(seed.value), false);
});

test('an image with no seed says so instead of dropping the row', () => {
  const [seed] = imageHeadlineFacts({});
  assert.equal(seed.value, '—');
  assert.equal(seed.copy, null);
});

test('the settings that decide the picture are published, folder-free', () => {
  const rows = imageSettingFacts(IMG);
  const by = Object.fromEntries(rows.map((r) => [r.key, r.value]));
  assert.equal(by.checkpoint, 'Ada-2500');
  assert.equal(by.base_model, 'zturbo');
  assert.equal(by.sampler, 'euler');
  assert.equal(by.cfg, '3.5');
  assert.equal(by.aspect, '9:16');
  assert.equal(by.extra_loras, 'Film @ 0.4');
  assert.equal(by.face_score, '0.679');
  assert.equal(by.created_at, '2026-07-27 10:11');
});

test('a legacy run that recorded nothing shows no rows rather than a table of dashes', () => {
  assert.deepEqual(imageSettingFacts({ id: 1, url: '/a.png' }), []);
});

test('unparseable extra LoRAs are an absent line, never a crash', () => {
  assert.equal(extraLoraSummary('{not json'), '');
  assert.equal(extraLoraSummary('[]'), '');
  assert.equal(extraLoraSummary(null), '');
  assert.equal(extraLoraSummary([{ filename: 'a\\b.safetensors' }]), 'b');
});

test('a checkpoint path is reduced to the file, without folder or extension', () => {
  assert.equal(checkpointFileLabel('z image\\Ada-2500.safetensors'), 'Ada-2500');
  assert.equal(checkpointFileLabel('flux/sub/Bob.ckpt'), 'Bob');
  assert.equal(checkpointFileLabel(''), '');
});

test('the prompt and its negative are last, as their own blocks', () => {
  assert.deepEqual(imagePromptBlocks(IMG).map((b) => b.key), ['prompt', 'negative']);
  assert.deepEqual(imagePromptBlocks({}), []);
});

test('a forty-line prompt opens folded; a one-line prompt never does', () => {
  const long = 'x'.repeat(PROMPT_FOLD_CHARS + 1);
  assert.equal(promptFold(long, false).collapsed, true);
  assert.equal(promptFold(long, true).collapsed, false);
  assert.equal(promptFold('a portrait', false).foldable, false);
  assert.equal(promptFold('a portrait', false).collapsed, false);
});

test('the tooltip line carries the facts and never the prompt', () => {
  const line = imageFactsLine(IMG);
  assert.equal(line, 'Step 2500 · Seed 208607443 · LoRA strength 0');
  assert.equal(line.includes('portrait'), false);
});

// A combined stack and an always-on style LoRA share the extra_loras column but
// are opposites to a user: co-stars chosen at their own weight vs a utility LoRA
// left on. One label for both said "Always-on LoRAs" over a combined run.
test('a combined stack is labelled as combined, not as always-on', () => {
  const stacked = JSON.stringify([
    { filename: 'z image\\lora_veldt.safetensors', strength: 0.55, combined: true },
    { filename: 'style\\Film.safetensors', strength: 0.4 },
  ]);
  const by = Object.fromEntries(
    imageSettingFacts({ extra_loras: stacked }).map((r) => [r.key, r.value]));
  assert.equal(by.combined_loras, 'lora_veldt @ 0.55');
  assert.equal(by.extra_loras, 'Film @ 0.4');
});

test('a run with no stack still shows only the always-on row', () => {
  const by = Object.fromEntries(
    imageSettingFacts({ extra_loras: '[{"filename":"style\\\\Film.safetensors","strength":0.4}]' })
      .map((r) => [r.key, r.value]));
  assert.equal(by.extra_loras, 'Film @ 0.4');
  assert.equal(by.combined_loras, undefined);
});

test('external entries have their own row and leave always-on', () => {
  const RAW = JSON.stringify([
    { filename: 'style.safetensors', strength: 0.5 },
    { filename: 'k\\member.safetensors', strength: 1, combined: true },
    { filename: 'detail.safetensors', strength: 0.7, external: true },
  ]);
  assert.match(extraLoraSummary(RAW, { only: 'external' }), /detail @ 0\.7/);
  const always = extraLoraSummary(RAW, { only: 'always-on' });
  assert.match(always, /style/);
  assert.doesNotMatch(always, /detail/);
  assert.doesNotMatch(always, /member/);
});

test('imageSettingFacts rows: external row present, keyed external_loras', () => {
  const RAW = JSON.stringify([
    { filename: 'style.safetensors', strength: 0.5 },
    { filename: 'k\\member.safetensors', strength: 1, combined: true },
    { filename: 'detail.safetensors', strength: 0.7, external: true },
  ]);
  const rows = imageSettingFacts({ extra_loras: RAW });
  const ext = rows.find((r) => r.key === 'external_loras');
  assert.ok(ext);
  assert.equal(ext.label, 'External LoRAs');
  const always = rows.find((r) => r.key === 'extra_loras');
  assert.doesNotMatch(always.value, /detail/);
});

