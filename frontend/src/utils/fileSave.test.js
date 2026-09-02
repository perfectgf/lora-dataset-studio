import test from 'node:test'
import assert from 'node:assert/strict'
import { saveUrlAsFile } from './fileSave.js'

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
