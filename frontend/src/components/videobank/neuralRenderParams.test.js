import test from 'node:test'
import assert from 'node:assert/strict'
import {
  NR_DEFAULTS, NR_PRESETS, TEMPORAL_MIN_WIDTH, normalizeNrParams, presetFor,
  temporalOutcome, nrRefusal, costMultiplier, neuralRenderTags,
} from './neuralRenderParams.js'

test('a render is remembered by its pushes, its mode and its cost', () => {
  assert.deepEqual(neuralRenderTags(null), [])
  assert.deepEqual(neuralRenderTags({ tone: 1, structure: 1, strength: 1, passes: 1, scale: 1, temporal: 'auto', temporal_used: false }), ['still'])
  assert.deepEqual(neuralRenderTags({ tone: 0, structure: 1.5, strength: 2.3, passes: 3, scale: 2, automask: true, temporal: 'auto', temporal_used: true, ms_per_frame: 261.8 }),
    ['tone 0', 'structure 1.5', 'strength ×2.3', '3 passes', '2× render', 'auto mask', 'temporal', '262 ms/frame'])
  // Before the child answered, the request stands in for the fact.
  assert.deepEqual(neuralRenderTags({ temporal: 'on' }), ['temporal'])
})

test('defaults are the photoreal preset, and the flat-art preset only moves tone', () => {
  assert.equal(presetFor(NR_DEFAULTS), 'photo')
  const flat = NR_PRESETS.find((p) => p.id === 'flat')
  assert.equal(flat.params.tone, 0)
  assert.equal(flat.params.structure, NR_DEFAULTS.structure)
  assert.equal(presetFor(flat.params), 'flat')
})

test('normalize clamps the two dials to 0..2, coerces the flag and falls back on junk', () => {
  assert.deepEqual(normalizeNrParams({ tone: 5, structure: -1, automask: 1, temporal: 'on' }),
    { tone: 2, structure: 0, automask: true, temporal: 'on', strength: 1, passes: 1, scale: 1 })
  assert.deepEqual(normalizeNrParams({ tone: 'abc', temporal: 'sideways' }),
    { ...NR_DEFAULTS })
  assert.deepEqual(normalizeNrParams(null), { ...NR_DEFAULTS })
})

test('the width floor is the measured one and auto falls back below it, on refuses', () => {
  assert.equal(TEMPORAL_MIN_WIDTH, 704)
  assert.match(temporalOutcome('auto', 512), /still mode/)
  assert.match(temporalOutcome('on', 512), /refused/)
  assert.equal(temporalOutcome('auto', 704), 'temporal mode')
  assert.equal(temporalOutcome('off', 4096), 'still mode')
  assert.match(temporalOutcome('auto', null), /per clip/)
})

test('the levers clamp to their measured ranges and price themselves', () => {
  const p = normalizeNrParams({ strength: 4, passes: 7, scale: '2' })
  assert.deepEqual([p.strength, p.passes, p.scale], [3, 3, 2])
  assert.deepEqual(normalizeNrParams({ strength: -1, passes: 0, scale: 3 }).strength, 0)
  assert.equal(normalizeNrParams({ passes: 2.5 }).passes, 1)
  assert.equal(normalizeNrParams({ strength: 1.55 }).strength, 1.6)
  assert.equal(costMultiplier({}), 1)
  assert.equal(costMultiplier({ passes: 3 }), 3)
  assert.equal(costMultiplier({ passes: 2, scale: 2 }), 8)
  assert.match(temporalOutcome('auto', 1024, 2), /extra passes/)
  assert.match(temporalOutcome('on', 1024, 2), /refused/)
  assert.equal(temporalOutcome('auto', 1024, 1), 'temporal mode')
})

test('the refusal sentence is built from the capability\'s own list', () => {
  assert.equal(nrRefusal({ ready: true, missing: [] }), null)
  assert.match(nrRefusal({ ready: false, missing: ['Windows — x', 'your own copy of nvngx_dlssnr.dll, placed in D'] }),
    /needs Windows — x; your own copy/)
  assert.match(nrRefusal(null), /checking/)
})
