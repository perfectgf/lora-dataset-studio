/* SeedVR2 settings — every dial reaches the user, or it does not exist.
 *
 * Requested by SurpassHR (GitHub #32) alongside the engine: "DiT/VAE model
 * locations, target resolution, batch size, etc.". A setting that lives in
 * config.py and nowhere else is invisible; one with no Guide entry is
 * undocumented; one with no help topic cannot be found by search. This test
 * pins the four surfaces together so the next dial cannot ship half-wired.
 *
 * node --test parses no JSX, so the card is read as TEXT — which is exactly the
 * granularity that matters here: the ids, the config keys and the reset targets.
 */
import test from 'node:test'
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict'

import { helpTopics } from '../src/help/helpRegistry.js'

const read = readSource
const CARD = read('src/components/settings/EnginesSection.jsx')
const GUIDE = read('../docs/guide/settings-reference.md')
const DEFAULTS = read('../backend/app/config.py')

// field → the DOM id of its control in the card.
const FIELDS = {
  model: 'seedvr2-model',
  vae: 'seedvr2-vae',
  resolution: 'seedvr2-resolution',
  max_resolution: 'seedvr2-max-resolution',
  color_correction: 'seedvr2-color',
  tiling: 'seedvr2-tiling',
  tile_px: 'seedvr2-tile-px',
  tile_threshold: 'seedvr2-tile-threshold',
  blocks_to_swap: 'seedvr2-swap',
}

test('every seedvr2 setting has a labelled control and a reset in the card', () => {
  for (const [field, domId] of Object.entries(FIELDS)) {
    assert.ok(CARD.includes(`id="${domId}"`), `${field}: no control id="${domId}"`)
    assert.ok(CARD.includes(`htmlFor="${domId}"`), `${field}: control has no <label>`)
    assert.ok(CARD.includes(`setField('seedvr2', '${field}'`),
      `${field}: the control writes to some other config key`)
    assert.ok(CARD.includes(`section="seedvr2" field="${field}"`),
      `${field}: no ResetToDefault — a dial you cannot undo is a trap`)
  }
})

test('the config defaults carry every field the card writes', () => {
  const block = DEFAULTS.match(/'seedvr2': \{([\s\S]*?)\n    \},/)
  assert.ok(block, "backend defaults have no 'seedvr2' block")
  for (const field of Object.keys(FIELDS)) {
    assert.match(block[1], new RegExp(`'${field}':`), `${field}: missing from DEFAULTS`)
  }
})

test('the new dials are documented and findable in Help', () => {
  // The three this wave added. The older ones predate the contract and are
  // covered by the guide check below on their config key alone.
  const topics = new Set(helpTopics.map((t) => t.id))
  for (const id of ['seedvr2.vae', 'seedvr2.tile_px', 'seedvr2.tile_threshold']) {
    assert.ok(topics.has(id), `${id}: no help topic — Help search cannot find it`)
  }
  for (const field of Object.keys(FIELDS)) {
    assert.ok(GUIDE.includes(`\`seedvr2.${field}\``),
      `seedvr2.${field}: absent from docs/guide/settings-reference.md`)
  }
})

test('the tile bounds mirrored in the card match the backend clamps', () => {
  const helper = read('../backend/app/services/seedvr2_helper.py')
  assert.match(helper, /TILE_PX_MIN, TILE_PX_MAX = 512, 2048/)
  assert.match(helper, /TILE_ABOVE_FACTOR = 1\.5/)
  assert.match(CARD, /const SEEDVR2_TILE_MIN = 512/)
  assert.match(CARD, /const SEEDVR2_TILE_MAX = 2048/)
  assert.match(CARD, /const SEEDVR2_TILE_ABOVE_FACTOR = 1\.5/)
})

test('the batch-size refusal is still stated, not silently dropped', () => {
  // The one item of #32 deliberately NOT shipped: batch_size is a temporal
  // window. Refusing it is defensible; refusing it without saying so is not.
  assert.match(CARD, /No batch size here, on purpose/)
  assert.match(GUIDE, /There is no batch-size setting, on purpose/)
})
