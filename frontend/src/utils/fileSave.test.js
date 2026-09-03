import test from 'node:test'
import assert from 'node:assert/strict'
import { isAbort, saveUrlAsFile } from './fileSave.js'

const res = (ok, { name = null, body = 'x', json = null } = {}) => ({
  ok,
  headers: { get: () => (name ? `attachment; filename="${name}"` : null) },
  blob: async () => body,
  json: async () => {
    if (json === null) throw new Error('not JSON')
    return json
  },
})

test('the saved file takes the name the server gave it', async () => {
  const saved = []
  const name = await saveUrlAsFile('/api/x', {
    fetchImpl: async () => res(true, { name: 'clip-42-vs-neural-45.mp4' }),
    saveBlob: (blob, n) => saved.push([blob, n]),
  })
  assert.equal(name, 'clip-42-vs-neural-45.mp4')
  assert.deepEqual(saved, [['x', 'clip-42-vs-neural-45.mp4']])
})

test('no Content-Disposition falls back to the caller’s name', async () => {
  const name = await saveUrlAsFile('/api/x', {
    fallbackName: 'comparison.mp4',
    fetchImpl: async () => res(true),
    saveBlob: () => {},
  })
  assert.equal(name, 'comparison.mp4')
})

test('a refusal throws the SERVER’s sentence, and saves nothing', async () => {
  let saves = 0
  await assert.rejects(
    () => saveUrlAsFile('/api/x', {
      fetchImpl: async () => res(false, { json: { error: 'this clip plays no render' } }),
      saveBlob: () => { saves += 1 },
    }),
    /this clip plays no render/)
  assert.equal(saves, 0, 'a failed request must not reach the disk')
})

test('a refusal that is not JSON still says something the user can read', async () => {
  await assert.rejects(
    () => saveUrlAsFile('/api/x', {
      failure: 'The comparison could not be built.',
      fetchImpl: async () => res(false),
      saveBlob: () => {},
    }),
    /The comparison could not be built/)
})

test('an abort is recognisable, so a closed layer shows no error', () => {
  const abort = Object.assign(new Error('The user aborted a request.'), { name: 'AbortError' })
  assert.equal(isAbort(abort), true)
  assert.equal(isAbort(new Error('this clip plays no render')), false)
  assert.equal(isAbort(null), false)
})

test('the DEFAULT transport carries the signal and the credentials', async () => {
  // The signal only does anything if the real fetch gets it, so this exercises
  // the default transport rather than an injected one — an injected fetchImpl
  // would prove the test's own stub, not the code.
  const controller = new AbortController()
  const original = globalThis.fetch
  let seenUrl = null
  let seenOpts = null
  globalThis.fetch = async (u, opts) => {
    seenUrl = u
    seenOpts = opts
    return res(true, { name: 'a.mp4' })
  }
  try {
    await saveUrlAsFile('/api/video-studio/clip/45/comparison', {
      signal: controller.signal,
      saveBlob: () => {},
    })
  } finally {
    globalThis.fetch = original
  }
  assert.equal(seenUrl, '/api/video-studio/clip/45/comparison')
  assert.equal(seenOpts.signal, controller.signal, 'the abort signal never reached fetch')
  assert.equal(seenOpts.credentials, 'same-origin')
})
