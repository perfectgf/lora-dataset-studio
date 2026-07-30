import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MAX_GENERATION_LORAS, MAX_GENERATION_LORA_PRESETS, KREA_LORA_STRENGTH_MAX,
  KREA_SLIDER_MAX, KREA_LORA_STRENGTH_DEFAULT, kreaStrengthRange,
  clampKreaLoraStrength, sanitizeKreaGenerationLoraPresets,
  kreaGenerationLoraPresetPayload,
} from './kreaGenerationLoras.js';

test('bypass filenames get the wide range, whatever the separator', () => {
  for (const name of ['krea2filterbypass3.safetensors', 'krea/filter_bypass.safetensors',
                      'FILTER-BYPASS.safetensors']) {
    assert.deepEqual(kreaStrengthRange(name), { min: 0, max: KREA_LORA_STRENGTH_MAX });
  }
});

test('ordinary filenames keep the narrow slider', () => {
  assert.deepEqual(kreaStrengthRange('krea/detail_slider.safetensors'),
    { min: 0, max: KREA_SLIDER_MAX });
  assert.deepEqual(kreaStrengthRange(undefined), { min: 0, max: KREA_SLIDER_MAX });
});

test('strength clamps to the server ceiling, never negative', () => {
  assert.equal(clampKreaLoraStrength(999), KREA_LORA_STRENGTH_MAX);
  assert.equal(clampKreaLoraStrength(-3), 0);
  assert.equal(clampKreaLoraStrength('abc'), 0);
  assert.equal(clampKreaLoraStrength(13), 13);
});

test('sanitizer drops junk, preserves order, applies the caps', () => {
  const out = sanitizeKreaGenerationLoraPresets([
    { name: '  Bypass  ', loras: [
      { file: ' krea/a.safetensors ', strength: 13 },
      { file: '', strength: 1 },
      { file: 'krea/b.safetensors', strength: 'x' },
      { file: 'krea/c.safetensors', strength: 999 },
    ] },
    { name: 'Bypass', loras: [] },
    { name: '   ', loras: [] },
    null,
  ]);
  assert.equal(out.length, 1);
  assert.equal(out[0].name, 'Bypass');
  assert.deepEqual(out[0].loras, [
    { file: 'krea/a.safetensors', strength: 13 },
    { file: 'krea/b.safetensors', strength: KREA_LORA_STRENGTH_DEFAULT },
    { file: 'krea/c.safetensors', strength: KREA_LORA_STRENGTH_MAX },
  ]);
});

test('caps bound rows and presets', () => {
  const rows = Array.from({ length: 30 }, (_, i) => ({ file: `k/${i}.safetensors`, strength: 1 }));
  const out = sanitizeKreaGenerationLoraPresets(
    Array.from({ length: 30 }, (_, i) => ({ name: `P${i}`, loras: rows })));
  assert.equal(out.length, MAX_GENERATION_LORA_PRESETS);
  assert.equal(out[0].loras.length, MAX_GENERATION_LORAS);
});

test('payload carries the name only when it would chain something', () => {
  const presets = [{ name: 'Bypass', loras: [{ file: 'k/a.safetensors', strength: 13 }] },
                   { name: 'Empty', loras: [] }];
  assert.deepEqual(
    kreaGenerationLoraPresetPayload({ isKrea: true, presetName: 'Bypass', presets }),
    { krea_generation_lora_preset: 'Bypass' });
  // No Krea in the run, no pick, an empty preset, an unknown name -> nothing sent.
  assert.deepEqual(kreaGenerationLoraPresetPayload({ isKrea: false, presetName: 'Bypass', presets }), {});
  assert.deepEqual(kreaGenerationLoraPresetPayload({ isKrea: true, presetName: '', presets }), {});
  assert.deepEqual(kreaGenerationLoraPresetPayload({ isKrea: true, presetName: 'Empty', presets }), {});
  assert.deepEqual(kreaGenerationLoraPresetPayload({ isKrea: true, presetName: 'Gone', presets }), {});
  assert.deepEqual(kreaGenerationLoraPresetPayload(), {});
});
