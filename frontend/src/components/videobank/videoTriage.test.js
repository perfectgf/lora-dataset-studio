import test from 'node:test'
import assert from 'node:assert/strict'

import {
  TRIAGE_STATUSES, toggleSelection, selectRange, triagePayload, triageAllPayload,
  triageAllConfirmation, STATUS_FILTERS, statusFilterCount, emptyGridMessage,
  hasMore,
} from "../../../../bundled/video/frontend/videobank/videoTriage.js"

// ---- THE footgun -------------------------------------------------------------

test('an empty selection produces NO body at all', () => {
  // `{ids: []}` means EVERY CLIP IN THE BANK server-side. The most ordinary
  // front-end bug there is — "the selection was empty and we posted anyway" —
  // would silently overwrite hundreds of decisions.
  assert.equal(triagePayload([], 'keep'), null)
  assert.equal(triagePayload(null, 'keep'), null)
  assert.equal(triagePayload(undefined, 'reject'), null)
  assert.equal(triagePayload([null, undefined], 'keep'), null)
})

test('"everything" is a SEPARATE function that says so on the wire', () => {
  assert.deepEqual(triageAllPayload('reject'), { ids: [], status: 'reject' })
  // Explicit `ids: []` rather than an omitted key: the request then states what
  // was meant, instead of relying on a default.
  assert.ok('ids' in triageAllPayload('keep'))
})

test('a selection posts exactly what was selected', () => {
  assert.deepEqual(triagePayload([4, 9], 'keep'), { ids: [4, 9], status: 'keep' })
})

test('a reject reason rides only on a reject', () => {
  assert.equal(triagePayload([1], 'reject', ' too dark ').reason, 'too dark')
  assert.ok(!('reason' in triagePayload([1], 'keep', 'too dark')))
  assert.ok(!('reason' in triagePayload([1], 'reject', '   ')))
})

test('an unknown status never reaches the server', () => {
  assert.equal(triagePayload([1], 'maybe'), null)
  assert.equal(triageAllPayload('maybe'), null)
  assert.deepEqual(TRIAGE_STATUSES, ['pending', 'keep', 'reject'])
})

test('the bulk confirmation says the count AND that off-screen shots are hit', () => {
  const msg = triageAllConfirmation('reject', 340)
  assert.match(msg, /ALL 340 shots/)
  assert.match(msg, /not currently on screen/)
  assert.match(triageAllConfirmation('keep', 1), /ALL 1 shot\b/)
})

// ---- selection ---------------------------------------------------------------

test('toggling adds then removes, keeping insertion order', () => {
  assert.deepEqual(toggleSelection([], 3), [3])
  assert.deepEqual(toggleSelection([3, 7], 9), [3, 7, 9])
  assert.deepEqual(toggleSelection([3, 7, 9], 7), [3, 9])
  assert.deepEqual(toggleSelection(null, 1), [1])
})

test('shift-range walks the VISIBLE order, in either direction', () => {
  // Ranging over the database order would select clips that are not on screen.
  const visible = [10, 20, 30, 40, 50]
  assert.deepEqual(selectRange([], visible, 20, 40), [20, 30, 40])
  assert.deepEqual(selectRange([], visible, 40, 20), [20, 30, 40])
  assert.deepEqual(selectRange([99], visible, 30, 30), [99, 30])
})

test('a range anchored outside the visible page degrades to a plain toggle', () => {
  assert.deepEqual(selectRange([], [10, 20], 999, 20), [20])
})

// ---- filters and empty states -------------------------------------------------

test('the chips carry their counts without a request each', () => {
  const counts = { clips: 340, pending: 12, keep: 128, reject: 200 }
  assert.equal(statusFilterCount(counts, 'all'), 340)
  assert.equal(statusFilterCount(counts, 'keep'), 128)
  assert.equal(statusFilterCount(counts, 'pending'), 12)
  assert.equal(statusFilterCount(null, 'keep'), 0)
  assert.deepEqual(STATUS_FILTERS.map((f) => f.key), ['all', 'pending', 'keep', 'reject'])
})

test('an empty grid explains WHY it is empty', () => {
  // "No results" on a bank that was never scanned reads as a broken app.
  assert.match(emptyGridMessage({ counts: { sources: 0 } }), /no files yet/)
  assert.match(emptyGridMessage({ counts: { sources: 12, clips: 0 } }), /Files → Cut a shot by hand/)
  assert.equal(emptyGridMessage({ status: 'keep', counts: { sources: 1, clips: 5 } }),
    'No kept shot in this bank.')
  assert.equal(emptyGridMessage({ status: 'pending', sourceName: 'a.mp4', counts: { sources: 1, clips: 5 } }),
    'No untriaged shot in a.mp4.')
  assert.equal(emptyGridMessage({ sourceName: 'a.mp4', counts: { sources: 1, clips: 5 } }),
    'No shot in a.mp4.')
})

test('there is more to load while fewer than total are held', () => {
  assert.equal(hasMore({ loaded: 200, total: 340 }), true)
  assert.equal(hasMore({ loaded: 340, total: 340 }), false)
  assert.equal(hasMore({ loaded: 0, total: 0 }), false)
})
