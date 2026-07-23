import test from 'node:test'
import assert from 'node:assert/strict'

import { PREVIEW_SLOTS, hiddenCount, previewSlots } from '../src/components/bank/bankPreview.js'

test('previewSlots always yields a fixed-width strip', () => {
  assert.equal(previewSlots([1, 2, 3, 4, 5]).length, PREVIEW_SLOTS)
  assert.deepEqual(previewSlots([7, 8]), [7, 8, null, null, null])
  // A backend that ever over-delivers must not stretch the card.
  assert.deepEqual(previewSlots([1, 2, 3, 4, 5, 6, 7]), [1, 2, 3, 4, 5])
})

test('previewSlots degrades to empty tiles when the bank has no images', () => {
  for (const empty of [[], undefined, null]) {
    assert.deepEqual(previewSlots(empty), [null, null, null, null, null])
  }
})

test('hiddenCount reports the images the strip leaves out', () => {
  assert.equal(hiddenCount(34, [1, 2, 3, 4, 5]), 29)
  assert.equal(hiddenCount(5, [1, 2, 3, 4, 5]), 0)   // nothing hidden, no badge
  assert.equal(hiddenCount(3, [1, 2, 3]), 0)
  assert.equal(hiddenCount(0, []), 0)
  assert.equal(hiddenCount(undefined, [1]), 0)
})
