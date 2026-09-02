import test from 'node:test'
import assert from 'node:assert/strict'
import {
  NR_DEFAULTS, NR_PRESETS, TEMPORAL_MIN_WIDTH, normalizeNrParams, presetFor,
  temporalOutcome, nrRefusal,
} from './neuralRenderParams.js'

test('defaults are the photoreal preset, and the flat-art preset only moves tone', () => {
  assert.equal(presetFor(NR_DEFAULTS), 'photo')
  const flat = NR_PRESETS.find((p) => p.id === 'flat')
  assert.equal(flat.params.tone, 0)
  assert.equal(flat.params.structure, NR_DEFAULTS.structure)
  assert.equal(presetFor(flat.params), 'flat')
})

test('normalize clamps the two dials to 0..2, coerces the flag and falls back on junk', () => {
  assert.deepEqual(normalizeNrParams({ tone: 5, structure: -1, automask: 1, temporal: 'on' }),
    { tone: 2, structure: 0, automask: true, temporal: 'on' })
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

test('the refusal sentence is built from the capability\'s own list', () => {
  assert.equal(nrRefusal({ ready: true, missing: [] }), null)
  assert.match(nrRefusal({ ready: false, missing: ['Windows — x', 'your own copy of nvngx_dlssnr.dll, placed in D'] }),
    /needs Windows — x; your own copy/)
  assert.match(nrRefusal(null), /checking/)
})
