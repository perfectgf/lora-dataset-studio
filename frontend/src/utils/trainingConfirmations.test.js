import assert from 'node:assert/strict'
import test from 'node:test'

import { runConfirmableTrainingRequest } from './trainingConfirmations.js'

test('continue accumulates caption overrides until success without repeating a request', async () => {
  const calls = []
  const request = async (options) => {
    calls.push({ ...options })
    if (!options.allowUncaptioned) return { ok: false, error: 'UNCAPTIONED: missing' }
    if (!options.allowCaptionQuality) return { ok: false, error: 'CAPTION_QUALITY: identical' }
    if (!options.allowCaptionMismatch) return { ok: false, error: 'MISMATCH_CAPTION: prose expected' }
    return { ok: true }
  }
  const markerToFlag = (error) => {
    if (error.startsWith('UNCAPTIONED: ')) return 'allow_uncaptioned'
    if (error.startsWith('CAPTION_QUALITY: ')) return 'allow_caption_quality'
    if (error.startsWith('MISMATCH_CAPTION: ')) return 'allow_caption_mismatch'
    return null
  }

  const result = await runConfirmableTrainingRequest(request, { masked: false }, markerToFlag)

  assert.equal(result.response.ok, true)
  assert.equal(calls.length, 4)
  assert.deepEqual(calls.map((options) => ({
    uncaptioned: !!options.allowUncaptioned,
    quality: !!options.allowCaptionQuality,
    mismatch: !!options.allowCaptionMismatch,
  })), [
    { uncaptioned: false, quality: false, mismatch: false },
    { uncaptioned: true, quality: false, mismatch: false },
    { uncaptioned: true, quality: true, mismatch: false },
    { uncaptioned: true, quality: true, mismatch: true },
  ])
})

test('unknown refusal cannot retry an unchanged request forever', async () => {
  let calls = 0
  const result = await runConfirmableTrainingRequest(
    async () => { calls += 1; return { ok: false, error: 'OTHER' } },
    {},
    () => 'unknown_flag',
  )
  assert.equal(result.response.ok, false)
  assert.equal(calls, 1)
})
