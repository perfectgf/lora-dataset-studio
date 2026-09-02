import test from 'node:test'
import assert from 'node:assert/strict'
import { readLock, writeLock } from './sliderLockStorage.js'

const fake = (initial = {}, { throws = false } = {}) => {
  const map = { ...initial }
  return {
    getItem(k) { if (throws) throw new Error('denied'); return k in map ? map[k] : null },
    setItem(k, v) { if (throws) throw new Error('denied'); map[k] = String(v) },
    dump: () => map,
  }
}

test('a slider nobody has touched is LOCKED', () => {
  // The default is the whole point: an unlocked-by-default dial guards nothing
  // on the one device where the accident happens.
  assert.equal(readLock('videoStudio.lock.length', fake()), true)
  assert.equal(readLock('videoStudio.lock.length', fake({ 'videoStudio.lock.length': 'false' })), false)
  assert.equal(readLock('videoStudio.lock.length', fake({ 'videoStudio.lock.length': 'true' })), true)
})

test('a store that refuses to answer still answers LOCKED', () => {
  // Private window, storage disabled, quota gone: the safe end of the guess is
  // the one that cannot move a value by itself.
  assert.equal(readLock('k', fake({}, { throws: true })), true)
  assert.equal(readLock('', fake({ '': 'false' })), true)
})

test('the choice is remembered per slider, not per app', () => {
  const store = fake()
  writeLock('videoStudio.lock.length', false, store)
  assert.equal(store.dump()['videoStudio.lock.length'], 'false')
  // Unlocking the length must not unlock the resolution: somebody working on
  // one dial has not asked for the others to be open.
  assert.equal(readLock('videoStudio.lock.megapixels', store), true)
})

test('a store that cannot write never breaks the toggle', () => {
  // The lock still opens for this page — it simply will not be there next time.
  assert.equal(writeLock('k', false, fake({}, { throws: true })), false)
  assert.equal(writeLock('', true, fake()), true)
})
