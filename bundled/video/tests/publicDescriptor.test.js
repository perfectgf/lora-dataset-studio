import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import descriptor from '../frontend/index.js'

const manifest = JSON.parse(await readFile(new URL('../plugin.json', import.meta.url)))

test('Video has its own routes and Studio while Live remains independent', () => {
  assert.equal(descriptor.id, 'video')
  assert.deepEqual(manifest.requires, [])
  assert.deepEqual(descriptor.routes.map(r => r.path), ['/video-bank', '/video-dataset/:id'])
  assert.deepEqual(descriptor.slots['studio.tab'].map(s => s.id), ['video'])
  assert.deepEqual(descriptor.slots['setup.card'].map(card => card.id), ['video-studio'])
})

test('every help topic belongs to Video; global Canvas and Live topics are absent', () => {
  const ids = descriptor.help.map(t => t.id)
  assert.equal(new Set(ids).size, ids.length)
  assert.deepEqual([...ids].sort(), [...manifest.owns.help_topics].sort())
  assert.ok(ids.includes('setup-video-studio'))
  assert.ok(ids.includes('video-studio-reference-model'))
  assert.ok(ids.includes('video-auto-continue'))
  assert.ok(!ids.includes('setup-dlss5-install'))
  assert.ok(ids.every(id => !/canvas|live|battle/.test(id)))
})

test('news and guide contributions exactly match their declared ownership', () => {
  assert.deepEqual(descriptor.whatsNew.map(n => n.id), manifest.owns.whats_new_ids)
  assert.equal(new Set(descriptor.whatsNew.map(n => n.id)).size, descriptor.whatsNew.length)
  assert.deepEqual(descriptor.guide.sections.map(s => `${s.chapter}#${s.anchor}`),
    manifest.guide_ownership.sections)
})
