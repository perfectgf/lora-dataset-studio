import assert from 'node:assert/strict'
import test from 'node:test'

import {
  partitionKleinImproveSelection,
  runSequentialKleinImprove,
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

test('bulk Klein orchestration is sequential and reports partial failures', async () => {
  let active = 0
  let maxActive = 0
  const calls = []
  const progress = []
  const improve = async (id) => {
    active += 1
    maxActive = Math.max(maxActive, active)
    calls.push(id)
    await new Promise((resolve) => setImmediate(resolve))
    active -= 1
    return id === 2 ? { ok: false, error: 'full' } : { ok: true, candidate_id: id + 10 }
  }
  const result = await runSequentialKleinImprove(
    [{ id: 1 }, { id: 2 }, { id: 3 }], improve, (state) => progress.push(state.done),
  )
  assert.deepEqual(calls, [1, 2, 3])
  assert.equal(maxActive, 1)
  assert.deepEqual(progress, [1, 2, 3])
  assert.deepEqual(result.succeeded.map(({ image }) => image.id), [1, 3])
  assert.deepEqual(result.failed.map(({ image }) => image.id), [2])
})
