import test from 'node:test'
import assert from 'node:assert/strict'
import { DRIFT_TOLERANCE_S, syncActions, sidesFor } from './videoSync.js'

test('a follower in step needs nothing', () => {
  const a = { currentTime: 3.0, paused: false, playbackRate: 1 }
  assert.deepEqual(syncActions(a, { ...a }), [])
  assert.deepEqual(syncActions(a, { ...a, currentTime: 3.0 + DRIFT_TOLERANCE_S / 2 }), [])
})

test('drift past the tolerance is a seek, and only then a play', () => {
  const leader = { currentTime: 10, paused: false, playbackRate: 1 }
  const follower = { currentTime: 9.5, paused: true, playbackRate: 1 }
  assert.deepEqual(syncActions(leader, follower), [{ type: 'seek', value: 10 }, { type: 'play' }])
})

test('pause mirrors, rate mirrors first', () => {
  const leader = { currentTime: 2, paused: true, playbackRate: 0.5 }
  const follower = { currentTime: 2, paused: false, playbackRate: 1 }
  assert.deepEqual(syncActions(leader, follower), [{ type: 'rate', value: 0.5 }, { type: 'pause' }])
})

test('missing snapshots are a no-op, never a throw', () => {
  assert.deepEqual(syncActions(null, { currentTime: 0, paused: true }), [])
  assert.deepEqual(syncActions({ currentTime: 0, paused: true }, undefined), [])
})

test('the sides keep their names when swapped', () => {
  assert.deepEqual(sidesFor(false).map((s) => s.key), ['original', 'render'])
  assert.deepEqual(sidesFor(true).map((s) => s.key), ['render', 'original'])
  assert.match(sidesFor(false)[1].label, /DLSS 5/)
})
