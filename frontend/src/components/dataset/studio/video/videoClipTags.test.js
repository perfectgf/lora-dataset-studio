import test from 'node:test'
import assert from 'node:assert/strict'
import { clipTags } from './videoClipTags.js'

test('clipTags carries the same facts as the summary, one per pill, no emoji', () => {
  const tags = clipTags({ eros: true, lora: 'h3/lds/j.safetensors', lora_strength: 1.3,
    turbo: true, sparse: 'max', latent_upscale: true, steps: 6, seed: 42 })
  assert.deepEqual(tags, ['10Eros base', 'j @ 1.3', 'turbo', 'sparse max', 'upscale ×2', '6 steps', 'seed 42'])
  assert.ok(tags.every((t) => !/\p{Extended_Pictographic}/u.test(t)))
  assert.deepEqual(clipTags({ steps: 30 }), ['no LoRA', '30 steps'])
  assert.deepEqual(clipTags(null), [])
})
