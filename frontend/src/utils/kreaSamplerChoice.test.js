/* Contract for the one Krea sampler menu that feeds two payload fields.

   The regression worth pinning: a preset name reaching the `sampler` field. That
   value is written straight onto the KSampler's `sampler_name`, where ComfyUI
   compares it to its own sampler list and refuses the ENTIRE graph — so the
   symptom is not "the preset was ignored", it is a run whose every tile failed,
   with nothing on screen naming the cause.

   The round-trip matters just as much: the menu has to be able to show back what
   a resumed run actually used, or picking a preset becomes a one-way door. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  KREA_PRESET_PREFIX, KREA_SAMPLER_PRESETS_FALLBACK,
  isPresetChoice, presetChoice, presetOf,
  splitSamplerChoice, joinSamplerChoice,
} from './kreaSamplerChoice.js';

test('a preset choice never reaches the sampler_name field', () => {
  for (const preset of KREA_SAMPLER_PRESETS_FALLBACK) {
    const split = splitSamplerChoice(presetChoice(preset));
    assert.equal(split.sampler, '', `${preset} leaked into sampler_name`);
    assert.equal(split.sampler_preset, preset);
  }
});

test('a stock sampler name never reaches the preset field', () => {
  const split = splitSamplerChoice('er_sde');
  assert.equal(split.sampler, 'er_sde');
  assert.equal(split.sampler_preset, '');
});

test('exactly one field is ever set', () => {
  for (const value of ['', 'euler', presetChoice('max'), undefined, null]) {
    const { sampler, sampler_preset: preset } = splitSamplerChoice(value);
    assert.ok(!(sampler && preset),
      `both fields set for ${JSON.stringify(value)} — the server would have to guess`);
  }
});

test('Auto sends neither field, leaving the workflow defaults alone', () => {
  assert.deepEqual(splitSamplerChoice(''), { sampler: '', sampler_preset: '' });
});

test('the menu value round-trips through the payload', () => {
  for (const value of ['', 'er_sde', 'dpmpp_2m', presetChoice('neutral'), presetChoice('max')]) {
    assert.equal(joinSamplerChoice(splitSamplerChoice(value)), value,
      `round trip lost ${JSON.stringify(value)}`);
  }
});

test('a run saved before presets existed still restores its sampler', () => {
  assert.equal(joinSamplerChoice({ sampler: 'deis' }), 'deis');
  assert.equal(joinSamplerChoice({}), '');
  assert.equal(joinSamplerChoice(), '');
});

test('the prefix keeps the two namespaces apart', () => {
  // The day ComfyUI ships a sampler literally called "balanced", the bare name
  // would be ambiguous. The prefixed one cannot be.
  assert.ok(!isPresetChoice('balanced'));
  assert.ok(isPresetChoice('preset:balanced'));
  assert.equal(presetOf('balanced'), '');
  assert.equal(presetOf('preset:balanced'), 'balanced');
  assert.equal(KREA_PRESET_PREFIX, 'preset:');
});

test('neutral is offered, because it is the A/B reference column', () => {
  assert.ok(KREA_SAMPLER_PRESETS_FALLBACK.includes('neutral'));
  assert.equal(KREA_SAMPLER_PRESETS_FALLBACK[0], 'neutral');
});
