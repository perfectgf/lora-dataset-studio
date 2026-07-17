import assert from 'node:assert/strict'
import test from 'node:test'

import {
  clampLoraStrength, optionalLoraPayload, sanitizeGenerationLoras,
  LORA_STRENGTH_MAX, MAX_GENERATION_LORAS,
} from './generationLoras.js'

test('clampLoraStrength bounds to [0, 1.5] and collapses junk to 0', () => {
  assert.equal(clampLoraStrength(0.6), 0.6)
  assert.equal(clampLoraStrength(5), LORA_STRENGTH_MAX)
  assert.equal(clampLoraStrength(-0.3), 0)
  assert.equal(clampLoraStrength('nope'), 0)
  assert.equal(clampLoraStrength(undefined), 0)
})

test('sanitizeGenerationLoras drops junk, normalizes and caps, keeping order', () => {
  const rows = sanitizeGenerationLoras([
    { file: ' klein/a.safetensors ', strength: 0.7, nsfw_only: 0 },
    { file: '', strength: 1 },                       // blank -> dropped
    'garbage', null,                                 // malformed -> dropped
    { file: 'klein/b.safetensors', strength: 'x', nsfw_only: true },  // junk strength -> 0.6
    { file: 'klein/c.safetensors', strength: 9 },    // clamped to 1.5
  ])
  assert.deepEqual(rows, [
    { file: 'klein/a.safetensors', strength: 0.7, nsfw_only: false },
    { file: 'klein/b.safetensors', strength: 0.6, nsfw_only: true },
    { file: 'klein/c.safetensors', strength: 1.5, nsfw_only: false },
  ])
})

test('sanitizeGenerationLoras caps the list at MAX_GENERATION_LORAS', () => {
  const many = Array.from({ length: MAX_GENERATION_LORAS + 4 },
    (_, i) => ({ file: `klein/l${i}.safetensors`, strength: 0.5 }))
  assert.equal(sanitizeGenerationLoras(many).length, MAX_GENERATION_LORAS)
})

test('no rows armed (the default) -> empty payload', () => {
  const rows = [
    { file: 'klein/a.safetensors', strength: 0.7, nsfw_only: false, on: false },
    { file: 'klein/b.safetensors', strength: 0.7, nsfw_only: true, on: false },
  ]
  assert.deepEqual(optionalLoraPayload({ isKlein: true, nsfwMode: true, rows }), {})
  assert.deepEqual(optionalLoraPayload(), {})
})

test('armed rows ride in LIST ORDER as {file, strength}', () => {
  const rows = [
    { file: 'klein/a.safetensors', strength: 0.7, nsfw_only: false, on: true },
    { file: 'klein/b.safetensors', strength: 0.4, nsfw_only: false, on: false },
    { file: 'klein/c.safetensors', strength: 1.0, nsfw_only: false, on: true },
  ]
  assert.deepEqual(optionalLoraPayload({ isKlein: true, rows }), {
    generation_loras: [
      { file: 'klein/a.safetensors', strength: 0.7 },
      { file: 'klein/c.safetensors', strength: 1.0 },
    ],
  })
})

test('nsfw_only rows are STRICTLY gated behind the NSFW toggle', () => {
  const rows = [
    { file: 'klein/tex.safetensors', strength: 0.7, nsfw_only: false, on: true },
    { file: 'klein/hot.safetensors', strength: 0.8, nsfw_only: true, on: true },
  ]
  // 🔞 mode off -> the nsfw_only row is absent (fail-closed), the plain one rides.
  assert.deepEqual(optionalLoraPayload({ isKlein: true, nsfwMode: false, rows }), {
    generation_loras: [{ file: 'klein/tex.safetensors', strength: 0.7 }],
  })
  assert.deepEqual(optionalLoraPayload({ isKlein: true, nsfwMode: true, rows }), {
    generation_loras: [
      { file: 'klein/tex.safetensors', strength: 0.7 },
      { file: 'klein/hot.safetensors', strength: 0.8 },
    ],
  })
})

test('API engines never emit the generation_loras key', () => {
  const rows = [{ file: 'klein/a.safetensors', strength: 1, nsfw_only: false, on: true }]
  assert.deepEqual(optionalLoraPayload({ isKlein: false, nsfwMode: true, rows }), {})
})

test('strength 0 means off — the row is omitted, not sent as 0', () => {
  const rows = [{ file: 'klein/a.safetensors', strength: 0, nsfw_only: false, on: true }]
  assert.deepEqual(optionalLoraPayload({ isKlein: true, rows }), {})
})

test('overshooting strengths are clamped to 1.5 in the payload', () => {
  const rows = [{ file: 'klein/a.safetensors', strength: 9, nsfw_only: false, on: true }]
  assert.deepEqual(optionalLoraPayload({ isKlein: true, rows }), {
    generation_loras: [{ file: 'klein/a.safetensors', strength: 1.5 }],
  })
})

test('payload is capped at MAX_GENERATION_LORAS armed rows', () => {
  const rows = Array.from({ length: MAX_GENERATION_LORAS + 3 },
    (_, i) => ({ file: `klein/l${i}.safetensors`, strength: 0.5, nsfw_only: false, on: true }))
  const out = optionalLoraPayload({ isKlein: true, rows })
  assert.equal(out.generation_loras.length, MAX_GENERATION_LORAS)
})
