import assert from 'node:assert/strict'
import test from 'node:test'

import {
  clampLoraStrength, generationLoraPresetPayload, sanitizeGenerationLoraPresets,
  LORA_STRENGTH_MAX, MAX_GENERATION_LORAS, MAX_GENERATION_LORA_PRESETS,
} from './generationLoras.js'

const PRESETS = [
  { name: 'Full stack', loras: [
    { file: 'klein/a.safetensors', strength: 0.6 },
    { file: 'klein/b.safetensors', strength: 0.8 },
  ] },
  { name: 'Empty one', loras: [] },
]

test('clampLoraStrength bounds to [0, 1.5] and collapses junk to 0', () => {
  assert.equal(clampLoraStrength(0.6), 0.6)
  assert.equal(clampLoraStrength(5), LORA_STRENGTH_MAX)
  assert.equal(clampLoraStrength(-0.3), 0)
  assert.equal(clampLoraStrength('nope'), 0)
  assert.equal(clampLoraStrength(undefined), 0)
})

test('sanitizeGenerationLoraPresets drops junk, dedupes names, keeps order', () => {
  const out = sanitizeGenerationLoraPresets([
    { name: '  ', loras: [{ file: 'klein/x.safetensors' }] },   // blank name -> dropped
    'garbage', null,                                            // malformed -> dropped
    { name: ' Big ', loras: [
      { file: '' }, 'junk',                                     // bad rows -> dropped
      { file: ' klein/a.safetensors ', strength: 'x' },         // junk strength -> 0.6
      { file: 'klein/b.safetensors', strength: 9 },             // clamped to 1.5
    ] },
    { name: 'Big', loras: [] },                                 // duplicate name -> dropped
  ])
  assert.deepEqual(out, [
    { name: 'Big', loras: [
      { file: 'klein/a.safetensors', strength: 0.6 },
      { file: 'klein/b.safetensors', strength: 1.5 },
    ] },
  ])
})

test('sanitizeGenerationLoraPresets applies both caps', () => {
  const bigPreset = {
    name: 'Big',
    loras: Array.from({ length: MAX_GENERATION_LORAS + 4 },
      (_, i) => ({ file: `klein/l${i}.safetensors`, strength: 0.5 })),
  }
  const many = Array.from({ length: MAX_GENERATION_LORA_PRESETS + 3 },
    (_, i) => ({ name: `P${i}`, loras: [] }))
  const out = sanitizeGenerationLoraPresets([bigPreset, ...many])
  assert.equal(out.length, MAX_GENERATION_LORA_PRESETS)
  assert.equal(out[0].loras.length, MAX_GENERATION_LORAS)
})

test('no preset picked (the "None" default) -> empty payload', () => {
  assert.deepEqual(generationLoraPresetPayload({ isKlein: true, presetName: '', presets: PRESETS }), {})
  assert.deepEqual(generationLoraPresetPayload(), {})
})

test('a picked preset rides as its NAME only — the backend owns the chain', () => {
  assert.deepEqual(
    generationLoraPresetPayload({ isKlein: true, presetName: 'Full stack', presets: PRESETS }),
    { generation_lora_preset: 'Full stack' })
})

test('API engines never emit the preset key', () => {
  assert.deepEqual(
    generationLoraPresetPayload({ isKlein: false, presetName: 'Full stack', presets: PRESETS }),
    {})
})

test('unknown or empty presets are not sent (fail-closed client-side too)', () => {
  assert.deepEqual(
    generationLoraPresetPayload({ isKlein: true, presetName: 'Ghost', presets: PRESETS }),
    {})
  assert.deepEqual(
    generationLoraPresetPayload({ isKlein: true, presetName: 'Empty one', presets: PRESETS }),
    {})
})
