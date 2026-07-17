import assert from 'node:assert/strict'
import test from 'node:test'

import { clampLoraStrength, optionalLoraPayload, LORA_STRENGTH_MAX } from './generationLoras.js'

test('clampLoraStrength bounds to [0, 1.5] and collapses junk to 0', () => {
  assert.equal(clampLoraStrength(0.6), 0.6)
  assert.equal(clampLoraStrength(5), LORA_STRENGTH_MAX)
  assert.equal(clampLoraStrength(-0.3), 0)
  assert.equal(clampLoraStrength('nope'), 0)
  assert.equal(clampLoraStrength(undefined), 0)
})

test('both slots off (the default) -> empty payload', () => {
  assert.deepEqual(optionalLoraPayload({ isKlein: true, nsfwMode: true }), {})
  assert.deepEqual(optionalLoraPayload(), {})
})

test('ultra_real rides SFW and NSFW runs alike once toggled on', () => {
  assert.deepEqual(
    optionalLoraPayload({ isKlein: true, ultraRealOn: true, ultraRealStrength: 0.7 }),
    { ultra_real_strength: 0.7 })
  assert.deepEqual(
    optionalLoraPayload({ isKlein: true, nsfwMode: true, ultraRealOn: true, ultraRealStrength: 0.7 }),
    { ultra_real_strength: 0.7 })
})

test('nsfw_anatomy is STRICTLY gated behind the NSFW toggle', () => {
  // Toggle on for the slot but 🔞 mode off -> key absent (fail-closed).
  assert.deepEqual(
    optionalLoraPayload({ isKlein: true, nsfwMode: false, nsfwLoraOn: true, nsfwLoraStrength: 0.8 }),
    {})
  assert.deepEqual(
    optionalLoraPayload({ isKlein: true, nsfwMode: true, nsfwLoraOn: true, nsfwLoraStrength: 0.8 }),
    { nsfw_lora_strength: 0.8 })
})

test('API engines never emit the optional-LoRA keys', () => {
  assert.deepEqual(
    optionalLoraPayload({ isKlein: false, nsfwMode: true, ultraRealOn: true, ultraRealStrength: 1, nsfwLoraOn: true, nsfwLoraStrength: 1 }),
    {})
})

test('strength 0 means off — the key is omitted, not sent as 0', () => {
  assert.deepEqual(
    optionalLoraPayload({ isKlein: true, ultraRealOn: true, ultraRealStrength: 0 }),
    {})
})

test('overshooting strengths are clamped to 1.5 in the payload', () => {
  assert.deepEqual(
    optionalLoraPayload({ isKlein: true, nsfwMode: true, ultraRealOn: true, ultraRealStrength: 9, nsfwLoraOn: true, nsfwLoraStrength: 2 }),
    { ultra_real_strength: 1.5, nsfw_lora_strength: 1.5 })
})
