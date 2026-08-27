import assert from 'node:assert/strict'
import { test } from 'node:test'

import { ensureLicenceAck, hasLicenceAck, licencePrompt, readAcks } from './licenceAck.js'

/** A storage fake with localStorage's shape and none of its moods. */
function memoryStorage(seed = {}) {
  const map = new Map(Object.entries(seed))
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
  }
}

const H3 = {
  target_profile: 'minimax_h3',
  licence_note: 'MiniMax H3 Community License grants NO rights in the EU…',
}
const WAN = { target_profile: 'wan22_14b', licence_note: null }

test('a target without a licence note never asks', () => {
  const storage = memoryStorage()
  let asked = 0
  const ok = ensureLicenceAck(WAN, { storage, confirmFn: () => { asked += 1; return false } })
  assert.equal(ok, true)
  assert.equal(asked, 0)
  assert.equal(hasLicenceAck(WAN, storage), true)
})

test('a licence-carrying target asks once, and a yes persists for the profile', () => {
  const storage = memoryStorage()
  let asked = 0
  const yes = () => { asked += 1; return true }

  assert.equal(hasLicenceAck(H3, storage), false)
  assert.equal(ensureLicenceAck(H3, { storage, confirmFn: yes }), true)
  assert.equal(asked, 1)

  // Second launch, same profile, different dataset row: no second question.
  const sibling = { ...H3 }
  assert.equal(ensureLicenceAck(sibling, { storage, confirmFn: yes }), true)
  assert.equal(asked, 1)
  assert.deepEqual(readAcks(storage), { minimax_h3: true })
})

test('a refusal blocks the launch and persists NOTHING — no is about now', () => {
  const storage = memoryStorage()
  assert.equal(ensureLicenceAck(H3, { storage, confirmFn: () => false }), false)
  assert.deepEqual(readAcks(storage), {})
  // Asked again on the next attempt, and a yes then still works.
  assert.equal(ensureLicenceAck(H3, { storage, confirmFn: () => true }), true)
})

test('the prompt carries the note itself, not a paraphrase of it', () => {
  const p = licencePrompt(H3)
  assert.ok(p.includes(H3.licence_note))
  assert.match(p, /confirm to continue/i)
})

test('a storage that throws reads as nothing acknowledged, never as a crash', () => {
  const broken = {
    getItem: () => { throw new Error('denied') },
    setItem: () => { throw new Error('denied') },
  }
  assert.equal(hasLicenceAck(H3, broken), false)
  // A yes still lets THIS launch through even though it cannot be remembered.
  assert.equal(ensureLicenceAck(H3, { storage: broken, confirmFn: () => true }), true)
})

test('junk in the store reads as nothing acknowledged', () => {
  const storage = memoryStorage({ 'videoLicenceAck.v1': '{not json' })
  assert.equal(hasLicenceAck(H3, storage), false)
})
