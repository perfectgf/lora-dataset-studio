import assert from 'node:assert/strict'
import test from 'node:test'

import {
  describeKleinImproveLaunch,
  kleinImproveBatchLabel,
  partitionKleinImproveSelection,
} from './kleinBulkImprove.js'

test('bulk Klein selection keeps only sources with no active improvement child', () => {
  const images = [
    { id: 1, filename: 'one.webp', status: 'keep' },
    { id: 2, filename: 'two.webp', status: 'keep' },
    { id: 3, filename: 'candidate.webp', status: 'pending', derivation_kind: 'klein_image_improve', parent_image_id: 2 },
    { id: 4, filename: 'rescue.webp', derivation_kind: 'small_image_source' },
    { id: 5, filename: null, status: 'pending' },
  ]
  const { eligible, excluded } = partitionKleinImproveSelection(images, [1, 2, 3, 4, 5, 999])
  assert.deepEqual(eligible.map((image) => image.id), [1])
  assert.deepEqual(excluded.map(({ id }) => id), [2, 3, 4, 5, 999])
  assert.match(excluded[0].reason, /pending review/)
})

test('the ✨ button reads its progress from the server activity, not a tab-local loop', () => {
  assert.equal(kleinImproveBatchLabel(null), null)
  assert.equal(kleinImproveBatchLabel({ kind: 'generate', done: 9, total: 60 }), null)
  assert.equal(kleinImproveBatchLabel({ kind: 'improve', done: 61, total: 250 }),
    '✨ Improving 61/250')
  assert.equal(kleinImproveBatchLabel({ kind: 'improve', done: 0, total: 0 }), '✨ Improving…')
  assert.equal(kleinImproveBatchLabel({ kind: 'improve', done: 3, total: 250, cancelling: true }),
    '✨ Stopping…')
})

test('launch wording states the server-side contract', () => {
  const message = describeKleinImproveLaunch({ queued: 250, skipped: 4 })
  assert.match(message, /250 image\(s\) in the background/)
  assert.match(message, /4 not eligible/)
  assert.match(message, /close this tab/)
  assert.doesNotMatch(describeKleinImproveLaunch({ queued: 3 }), /not eligible/)
})
