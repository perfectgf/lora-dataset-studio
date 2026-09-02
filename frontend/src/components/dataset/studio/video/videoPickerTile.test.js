import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

import {
  clampTile, gridBoxHeight, readTile, writeTile,
  TILE_DEFAULT, TILE_MAX, TILE_MIN, TILE_STEP, TILE_STORAGE_KEY,
} from './videoPickerTile.js'

test('a tile size is numeric, snapped to the step and inside the dial', () => {
  assert.equal(clampTile(84), 84)
  assert.equal(clampTile('120'), 120)
  assert.equal(clampTile(85), 84)
  assert.equal(clampTile(86), 88)
  assert.equal(clampTile(10), TILE_MIN)
  assert.equal(clampTile(9999), TILE_MAX)
  assert.equal(clampTile(-1), TILE_MIN)
  for (const junk of ['', 'big', NaN, undefined, null, {}, Infinity]) {
    assert.equal(clampTile(junk), TILE_DEFAULT, `clampTile(${String(junk)})`)
  }
  assert.equal(TILE_DEFAULT % TILE_STEP, 0, 'the default must sit on the dial')
  assert.ok(TILE_MIN < TILE_DEFAULT && TILE_DEFAULT < TILE_MAX)
})

test('the stored size is read back clamped, and a missing or broken store means the default', () => {
  const store = new Map()
  const local = { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, v) }
  assert.equal(readTile(local), TILE_DEFAULT)
  writeTile(200, local)
  assert.equal(store.get(TILE_STORAGE_KEY), '200')
  assert.equal(readTile(local), 200)
  writeTile(5000, local)
  assert.equal(store.get(TILE_STORAGE_KEY), String(TILE_MAX), 'written clamped, so a later reader needs no clamp')
  assert.equal(readTile(local), TILE_MAX)
  store.set(TILE_STORAGE_KEY, 'garbage')
  assert.equal(readTile(local), TILE_DEFAULT)
  store.set(TILE_STORAGE_KEY, '')
  assert.equal(readTile(local), TILE_DEFAULT)
  const throwing = { getItem() { throw new Error('blocked') }, setItem() { throw new Error('quota') } }
  assert.equal(readTile(throwing), TILE_DEFAULT)
  assert.doesNotThrow(() => writeTile(120, throwing))
  assert.equal(readTile(null), TILE_DEFAULT)
})

test('with no store named, the helper finds the browser\'s — and survives one that throws on access', () => {
  // node has no localStorage: the default store is nothing, and nothing is fine.
  assert.equal(readTile(), TILE_DEFAULT)
  assert.doesNotThrow(() => writeTile(120))
  // A browser that blocks site data throws on the ACCESS itself, before any
  // getItem — the case that took the picker down while it rendered.
  const had = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true, get() { throw new Error('access is denied for this document') },
  })
  try {
    assert.equal(readTile(), TILE_DEFAULT)
    assert.doesNotThrow(() => writeTile(120))
  } finally {
    if (had) Object.defineProperty(globalThis, 'localStorage', had)
    else delete globalThis.localStorage
  }
})

test('the dial keeps the range and step of the concept sources\u2019 \ud83d\udd0d, so the two feel like one', () => {
  // The other side carries its numbers as literals in JSX; read both, or the
  // two dials drift apart with every gate green and the docstring quietly false.
  const concept = fs.readFileSync(new URL('../../ConceptSourcesPanel.jsx', import.meta.url), 'utf8')
  assert.match(concept, new RegExp(`<input type="range" min="${TILE_MIN}" max="${TILE_MAX}" step="${TILE_STEP}"`),
    'the two Preview size dials must keep the same range and step')
})

test('the scrolling box grows with the tile, between the old 288 px and 640 px', () => {
  assert.equal(gridBoxHeight(TILE_DEFAULT), 288, 'the default is at least the height the grids had')
  assert.equal(gridBoxHeight(TILE_MIN), 288)
  assert.equal(gridBoxHeight(160), 408)
  assert.equal(gridBoxHeight(TILE_MAX), 640)
  assert.equal(gridBoxHeight('nonsense'), 288)
  let prev = 0
  for (let t = TILE_MIN; t <= TILE_MAX; t += TILE_STEP) {
    const h = gridBoxHeight(t)
    assert.ok(h >= prev, `not monotonic at ${t}`)
    prev = h
  }
})
